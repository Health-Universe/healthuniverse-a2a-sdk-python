"""AsyncAgent for long-running tasks"""

import asyncio
import logging
import os
import uuid
from typing import Any

from a2a.types import Message, Part, Role, TaskState, TextPart

from health_universe_a2a.base import A2AAgent
from health_universe_a2a.context import AsyncContext, MessageContext
from health_universe_a2a.types.extensions import (
    BACKGROUND_JOB_EXTENSION_URI,
    FILE_ACCESS_EXTENSION_URI,
    AgentExtension,
)
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
)
from health_universe_a2a.update_client import BackgroundUpdateClient

logger = logging.getLogger(__name__)


class AsyncAgent(A2AAgent[AsyncContext]):
    """
    Agent for long-running tasks (hours) with async job processing.

    **When to Use AsyncAgent:**
    ✓ Task takes more than 5 minutes
    ✓ Task might run for hours (up to max_duration_seconds)
    ✓ Users don't need to wait for completion
    ✓ You want updates persisted to backend via POST
    ✓ SSE timeout limits would be exceeded

    **Use StreamingAgent instead if:**
    ✗ Task completes in under 5 minutes
    ✗ Users expect real-time streaming feedback

    Key differences from StreamingAgent:
    - Validation happens BEFORE job is enqueued
    - Returns immediately after validation with "submitted" status
    - Processing happens asynchronously
    - Updates POSTed to backend via /a2a/task-updates endpoint
    - No SSE streaming timeout constraints

    The agent automatically:
    - Declares background job extension support
    - Configures file access extension if requires_file_access() returns True
    - POSTs progress updates to backend
    - Handles job lifecycle (validation, enqueueing, async processing)

    Example:
        from health_universe_a2a import AsyncAgent, AsyncContext
        import json

        class HeavyProcessorAgent(AsyncAgent):
            def get_agent_name(self) -> str:
                return "Heavy Processor"

            def get_agent_description(self) -> str:
                return "Processes large datasets in background"

            def requires_file_access(self) -> bool:
                return True

            def get_max_duration_seconds(self) -> int:
                return 7200  # 2 hours

            async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
                if "file_uri" not in message:
                    return ValidationRejected(reason="No file URI provided")
                return ValidationAccepted(estimated_duration_seconds=3600)

            async def process_message(
                self, message: str, context: AsyncContext
            ) -> str:
                await context.update_progress("Loading file...", 0.1)
                data = await self.load_file(context.file_access_token)

                batches = self.split_batches(data, 10)

                for i, batch in enumerate(batches):
                    await context.update_progress(
                        f"Processing batch {i+1}/{len(batches)}",
                        progress=(i+1)/len(batches)
                    )

                    result = await self.process_batch(batch)

                    await context.add_artifact(
                        name=f"Batch {i+1} Results",
                        content=json.dumps(result),
                        data_type="application/json"
                    )

                return f"Processed {len(batches)} batches successfully!"

    Implementation Details:
        - Inherits from A2AAgent and implements AgentExecutor interface
        - Integrates with a2a-sdk's DefaultRequestHandler and TaskUpdater
        - Uses BackgroundUpdateClient for POSTing updates to Health Universe backend
        - SSE connection closes after validation/ack (final=True)
        - Background processing happens via asyncio.create_task()
        - HTTP client lifecycle managed in _run_background_work() with proper cleanup

    Background Processing Model:
        AsyncAgent uses a dual-phase approach to enable long-running tasks:

        **Phase 1: Validation & Acknowledgment (SSE)**
        1. Request arrives → validate_message() runs immediately
        2. If rejected: Error sent via SSE, request ends
        3. If accepted: "Job submitted" message sent via SSE with final=True
        4. SSE connection closes immediately after ack
        5. User receives confirmation and can close browser

        **Phase 2: Background Processing (POST)**
        1. asyncio.create_task() launches _run_background_work() asynchronously
        2. Task runs independently (not awaited by request handler)
        3. All progress updates sent via POST to /a2a/task-updates endpoint
        4. Updates persist in database, survives server restarts
        5. On completion: Final POST with results
        6. On error: POST with error details

        **Why asyncio.create_task()?**
        - Creates independent async task that doesn't block request handler
        - Task continues running after SSE connection closes
        - Request handler can return immediately
        - User doesn't wait for task completion

        **Connection Lifecycle:**
        - Request: Client → Agent (SSE established)
        - Validation: Agent validates message
        - Ack: Agent sends "submitted" + closes SSE (final=True)
        - Background: Task runs independently, POSTs updates
        - No timeout: Task can run for hours without connection constraints

    Server Shutdown & Cancellation:
        **Graceful Shutdown:**
        - Background tasks created with asyncio.create_task() continue running
        - If server shuts down: Tasks are cancelled by asyncio event loop
        - For production: Use proper task tracking and cleanup handlers
        - Consider: Persistent task queue (Celery, RQ) for critical workloads

        **Cancellation Handling:**
        AsyncAgent doesn't automatically handle cancellation during shutdown.
        For cancellation support, check context.is_cancelled() periodically:

        Example:
            async def process_message(self, message: str, context: AsyncContext) -> str:
                batches = self.split_data(message, batch_size=100)

                for i, batch in enumerate(batches):
                    # Check cancellation before each batch
                    if context.is_cancelled():
                        await context.update_progress("Cancelled by user", 1.0)
                        return f"Processing cancelled after {i} batches"

                    await self.process_batch(batch)
                    await context.update_progress(
                        f"Processed {i+1}/{len(batches)} batches",
                        (i+1)/len(batches)
                    )

                return "All batches processed successfully"

        **Production Recommendations:**
        1. Implement cancellation checks for long-running loops
        2. Save intermediate results periodically (checkpoint pattern)
        3. Use proper error handling with try/except blocks
        4. Log progress for debugging and monitoring
        5. Consider using persistent queue systems for critical tasks
        6. Set reasonable max_duration_seconds to prevent runaway tasks
    """

    def supports_streaming(self) -> bool:
        """Disable SSE streaming (uses POST updates instead)."""
        return False

    def supports_push_notifications(self) -> bool:
        """Enable push notification support."""
        return True

    def get_extensions(self) -> list[AgentExtension]:
        """Auto-configure extensions."""
        extensions = [
            # Always add background job extension
            AgentExtension(uri=BACKGROUND_JOB_EXTENSION_URI)
        ]

        # Add file access if needed
        if self.requires_file_access():
            extensions.append(AgentExtension(uri=FILE_ACCESS_EXTENSION_URI))

        return extensions

    def get_max_duration_seconds(self) -> int:
        """
        Maximum duration for background task.

        Override to set custom timeout (default: 1 hour).

        Returns:
            Max duration in seconds
        """
        return 3600

    async def handle_request(
        self, message: str, context: MessageContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle async agent request: validate → ack via SSE → process with POST updates.

        For AsyncAgent, this method:
        1. Validates using immediate_updater (SSE)
        2. If rejected: Sends error via SSE and returns
        3. If accepted:
           - Sends ack via SSE (immediate_updater)
           - Creates background_updater (POST)
           - Immediately calls process_message() with background context
           - All agent updates POSTed, not via SSE

        The dual-updater pattern:
        - immediate_updater (context._updater): For validation/ack via SSE
        - background_updater (async_context._update_client): For processing via POST

        Args:
            message: The message to process
            context: Message context with immediate_updater (SSE)
            metadata: Request metadata for validation

        Returns:
            Final result from process_message() if validation passed, None if rejected
        """
        # Step 1: Validate with immediate_updater (SSE)
        validation_result = await self.validate_message(message, metadata)

        # Step 2: Handle rejection via SSE
        if isinstance(validation_result, ValidationRejected):
            self.logger.warning(f"Message validation failed: {validation_result.reason}")
            if context._updater:
                text_part = TextPart(text=f"Validation failed: {validation_result.reason}")
                msg = Message(
                    message_id=str(uuid.uuid4()), role=Role.agent, parts=[Part(root=text_part)]
                )
                await context._updater.reject(message=msg)
            return None

        # Step 3: Handle acceptance
        if isinstance(validation_result, ValidationAccepted):
            duration_msg = ""
            if validation_result.estimated_duration_seconds:
                minutes = validation_result.estimated_duration_seconds // 60
                duration_msg = (
                    f" (estimated: {minutes} min)"
                    if minutes > 0
                    else f" (estimated: {validation_result.estimated_duration_seconds}s)"
                )

            ack_message = f"Job submitted successfully{duration_msg}. Processing will continue in the background."

            self.logger.info("Validation passed, sending ack via SSE")

            # Send ack via immediate_updater (SSE)
            if context._updater:
                ack_metadata = {}
                if validation_result.estimated_duration_seconds:
                    ack_metadata["estimated_duration_seconds"] = (
                        validation_result.estimated_duration_seconds
                    )

                text_part = TextPart(text=ack_message)
                msg = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(root=text_part)],
                    metadata=ack_metadata if ack_metadata else None,
                )
                # Send submitted status and close SSE connection (final=True)
                # Background processing will continue with POST updates
                await context._updater.update_status(
                    state=TaskState.submitted,
                    message=msg,
                    final=True,
                    metadata=ack_metadata if ack_metadata else None,
                )

            # Step 4: Extract background job params
            if BACKGROUND_JOB_EXTENSION_URI not in metadata:
                self.logger.error("Background job extension missing - cannot process async")
                return ack_message

            from health_universe_a2a.types.extensions import BackgroundJobExtensionParams

            bg_params = BackgroundJobExtensionParams.model_validate(
                metadata[BACKGROUND_JOB_EXTENSION_URI]
            )
            job_id = bg_params.job_id
            api_key = bg_params.api_key

            # Step 5: Launch background task (don't await!)
            # The SSE connection will close immediately after returning ack_message
            # Background processing continues with POST updates
            self.logger.info(
                f"Launching background task for job_id={job_id}, SSE will close after ack"
            )

            asyncio.create_task(
                self._run_background_work(
                    message=message,
                    job_id=job_id,
                    api_key=api_key,
                    metadata=metadata,
                    user_id=context.user_id,
                    thread_id=context.thread_id,
                    file_access_token=context.file_access_token,
                )
            )

            # Return immediately - SSE connection closes, background processing continues
            return ack_message

        return None

    # The actual execute() method would integrate with your A2A SDK here
    # This would handle:
    # 1. Extracting message from RequestContext
    # 2. Running validation BEFORE queueing
    # 3. If rejected, returning rejection immediately
    # 4. If accepted, returning "submitted" status
    # 5. Enqueueing background work
    # 6. Background work POSTs updates as it progresses

    async def _run_background_work(
        self,
        message: str,
        job_id: str,
        api_key: str,
        metadata: dict[str, Any],
        user_id: str | None = None,
        thread_id: str | None = None,
        file_access_token: str | None = None,
    ) -> None:
        """
        Run background work with POST updates.

        This method creates the HTTP client and async context, processes the message,
        and ensures proper cleanup. It's designed to be called with asyncio.create_task()
        so the SSE connection can close immediately after ack.

        Args:
            message: The message to process
            job_id: Background job ID
            api_key: API key for POSTing updates to Health Universe backend
            metadata: Request metadata
            user_id: Optional user ID from request
            thread_id: Optional thread ID from request
            file_access_token: Optional file access token from extensions
        """
        # Create update client (owns HTTP connection)
        base_url = os.getenv("BACKGROUND_UPDATE_URL", "https://api.healthuniverse.com")
        update_client = BackgroundUpdateClient(
            api_key=api_key,
            job_id=job_id,
            base_url=base_url,
        )

        try:
            # Build async context with POST updater
            async_context = AsyncContext(
                user_id=user_id,
                thread_id=thread_id,
                file_access_token=file_access_token,
                metadata=metadata,
                job_id=job_id,
                _update_client=update_client,
            )

            # Process message with POST updates
            self.logger.info(f"Background processing started for job {job_id}")
            final_message = await self.process_message(message, async_context)

            # POST completion
            self.logger.info(f"Background processing completed for job {job_id}")
            await update_client.post_completion(final_message)

        except Exception as e:
            self.logger.error(f"Background work failed for job {job_id}: {e}", exc_info=True)
            await update_client.post_failure(str(e))

        finally:
            # Always close HTTP client to prevent connection leaks
            await update_client.close()
