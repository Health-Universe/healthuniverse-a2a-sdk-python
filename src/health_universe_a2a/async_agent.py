"""AsyncAgent for long-running tasks"""

import logging
import os
from typing import Any

from a2a.types import Message, Part, Role, TextPart

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


class AsyncAgent(A2AAgent):
    """
    Agent for long-running tasks (hours) with async job processing.

    Use this agent type when:
    - Processing takes more than a few minutes
    - Users don't need to wait for completion
    - You want to POST updates to backend for persistence
    - Task duration exceeds typical HTTP/SSE timeout limits

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

    Note:
        This is a stub implementation. The actual integration with the A2A SDK
        (AgentExecutor, EventQueue, etc.) should be implemented based on your
        specific A2A framework.
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
                msg = Message(role=Role.agent, parts=[Part(root=text_part)])
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
                    role=Role.agent,
                    parts=[Part(root=text_part)],
                    metadata=ack_metadata if ack_metadata else None,
                )
                await context._updater.submit(message=msg)

            # Step 4: Extract background job params and create POST updater
            if BACKGROUND_JOB_EXTENSION_URI not in metadata:
                self.logger.error("Background job extension missing - cannot process async")
                return ack_message

            from health_universe_a2a.types.extensions import BackgroundJobExtensionParams

            bg_params = BackgroundJobExtensionParams.model_validate(
                metadata[BACKGROUND_JOB_EXTENSION_URI]
            )
            job_id = bg_params.job_id
            api_key = bg_params.api_key

            # Create background update client (POSTs to /a2a/task-updates)
            update_client = BackgroundUpdateClient(
                api_key=api_key,
                job_id=job_id,
                base_url=os.getenv("BACKGROUND_UPDATE_URL", "https://api.healthuniverse.com"),
            )

            # Create async context with background updater (POST)
            async_context = AsyncContext(
                user_id=context.user_id,
                thread_id=context.thread_id,
                file_access_token=context.file_access_token,
                metadata=context.metadata,
                job_id=job_id,
                _update_client=update_client,
            )

            self.logger.info(f"Starting async processing (job_id={job_id}), updates via POST")

            # Step 5: Call lifecycle hooks and process immediately
            await self.on_task_start(message, async_context)

            try:
                # Process immediately with background updater (POST)
                result = await self.process_message(message, async_context)

                # Call completion hook
                await self.on_task_complete(message, result, async_context)

                return result

            except Exception as error:
                # Call error hook
                custom_error = await self.on_task_error(message, error, async_context)
                error_msg = custom_error or f"Processing error: {str(error)}"

                # Send error via POST
                if async_context._update_client:
                    await async_context._update_client.post_update(
                        update_type="progress",
                        status_message=error_msg,
                        task_status="failed",
                    )

                raise

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
        context: AsyncContext,
        update_client: BackgroundUpdateClient,
    ) -> None:
        """
        Run async work and POST updates.

        Args:
            message: The message to process
            context: Background context with job info
            update_client: Client for POSTing updates
        """
        try:
            # Call agent's process_message
            final_message = await self.process_message(message, context)

            # POST completion
            await update_client.post_completion(final_message)

        except Exception as e:
            self.logger.error(f"Background work failed for job {context.job_id}: {e}")
            await update_client.post_failure(str(e))

        finally:
            await update_client.close()

    def _build_background_context(
        self, request_context: Any, job_id: str, api_key: str
    ) -> AsyncContext:
        """
        Build async context from A2A request context.

        Args:
            request_context: A2A RequestContext
            job_id: Async job ID
            api_key: API key for POSTing updates

        Returns:
            AsyncContext for agent use
        """
        # This would extract metadata from your A2A SDK's RequestContext
        # Placeholder implementation:
        metadata: dict[str, Any] = {}
        file_access_token = None

        # Extract file access token if present
        if FILE_ACCESS_EXTENSION_URI in metadata:
            file_access_token = metadata[FILE_ACCESS_EXTENSION_URI].get("access_token")

        # Create update client
        base_url = os.getenv("HU_LANGGRAPH_URL", "http://localhost:8000")
        update_client = BackgroundUpdateClient(job_id=job_id, api_key=api_key, base_url=base_url)

        return AsyncContext(
            job_id=job_id,
            user_id=metadata.get("user_id"),
            thread_id=metadata.get("thread_id"),
            file_access_token=file_access_token,
            metadata=metadata,
            _update_client=update_client,
            _request_context=request_context,
        )
