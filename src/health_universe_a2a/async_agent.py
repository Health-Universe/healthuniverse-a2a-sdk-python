"""AsyncAgent for long-running tasks"""

import asyncio
import logging
import os
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from a2a.types import AgentExtension, Message, Part, Role, TaskState, TextPart

from health_universe_a2a.base import A2AAgentBase
from health_universe_a2a.context import BackgroundContext, StreamingContext
from health_universe_a2a.types.extensions import (
    BACKGROUND_JOB_EXTENSION_URI,
    FILE_ACCESS_EXTENSION_URI,
    HU_LOG_LEVEL_EXTENSION_URI,
)
from health_universe_a2a.types.validation import ValidationRejected
from health_universe_a2a.update_client import (
    BackgroundUpdateClient,
)

if TYPE_CHECKING:
    from health_universe_a2a.inspect_ai.logger import InspectLogger

logger = logging.getLogger(__name__)


class AsyncAgent(A2AAgentBase):
    """
    Agent for Health Universe - the primary agent class.

    Use this class (or the Agent alias) for all Health Universe agents.
    All agents run asynchronously with progress updates persisted to the database.

    **Key Features:**
    - Automatic file access (context.documents provides document operations)
    - Progress updates stored in database (queryable for billing, analytics)
    - Long-running tasks (up to max_duration_seconds, default 1 hour)
    - Inter-agent communication via call_agent() or call_other_agent()

    **Required Methods:**
    - get_agent_name(): Return the agent's display name
    - get_agent_description(): Return what the agent does
    - process_message(message, context): Process messages and return results

    **Optional Methods:**
    - validate_message(message, metadata): Validate before processing
    - get_max_duration_seconds(): Override max task duration (default: 1 hour)

    Example:
        from health_universe_a2a import Agent, AgentContext
        import json

        class DataAnalyzer(Agent):
            def get_agent_name(self) -> str:
                return "Clinical Data Analyzer"

            def get_agent_description(self) -> str:
                return "Analyzes clinical datasets and generates insights"

            async def process_message(self, message: str, context: AgentContext) -> str:
                # List documents in the thread
                docs = await context.documents.list_documents()

                # Find and read a specific document
                for doc in docs:
                    if "protocol" in doc.name.lower():
                        content = await context.documents.download_text(doc.id)
                        break

                # Process with progress updates
                await context.update_progress("Analyzing data...", 0.5)
                results = await self.analyze(content)

                # Write results as a new document
                await context.documents.write(
                    "Analysis Results",
                    json.dumps(results, indent=2),
                    filename="analysis.json"
                )

                return f"Analysis complete! Found {len(results)} insights."

        if __name__ == "__main__":
            agent = DataAnalyzer()
            agent.serve()

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
            async def process_message(self, message: str, context: BackgroundContext) -> str:
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
        """
        Auto-configure extensions for Health Universe agents.

        All agents automatically get:
        - Background job extension (for async processing)
        - File access v2 extension (for document operations)
        - Log level extension (for update importance filtering)

        This enables document operations without any configuration:
            docs = await context.document_client.list_documents()
            content = await context.document_client.download_text(doc_id)
            await context.document_client.write("Results", json.dumps(data))
        """
        return [
            AgentExtension(uri=BACKGROUND_JOB_EXTENSION_URI),
            AgentExtension(uri=FILE_ACCESS_EXTENSION_URI),
            AgentExtension(uri=HU_LOG_LEVEL_EXTENSION_URI),
        ]

    def get_max_duration_seconds(self) -> int:
        """
        Maximum duration for background task.

        Override to set custom timeout (default: 1 hour).

        Returns:
            Max duration in seconds
        """
        return 3600

    @abstractmethod
    async def process_message(self, message: str, context: BackgroundContext) -> str:
        """
        Process a message with background context.

        This method is called in the background after validation passes and
        the SSE connection closes. All updates are sent via POST to the backend.

        Args:
            message: The message to process
            context: BackgroundContext with update_client for POSTing updates

        Returns:
            Final result message

        Example:
            async def process_message(self, message: str, context: BackgroundContext) -> str:
                await context.update_progress("Starting...", 0.1)
                result = await self.do_work(message)
                await context.add_artifact("Results", json.dumps(result), "application/json")
                return "Processing complete!"
        """
        pass

    async def on_task_start(self, message: str, context: BackgroundContext) -> None:  # noqa: B027
        """
        Called before process_message starts in the background.

        This is called AFTER the SSE connection has closed, during the background
        processing phase.

        Use for: logging, metrics, setup

        Args:
            message: The message being processed
            context: BackgroundContext (SSE closed, using POST updates)

        Example:
            async def on_task_start(self, message: str, context: BackgroundContext) -> None:
                self.logger.info(f"Starting background task for job {context.job_id}")
                await self.metrics.increment("background_tasks_started")
        """
        pass

    async def on_task_complete(self, message: str, result: str, context: BackgroundContext) -> None:  # noqa: B027
        """
        Called after process_message completes successfully in the background.

        This is called AFTER the SSE connection has closed, during the background
        processing phase.

        Use for: logging, metrics, cleanup

        Args:
            message: The message that was processed
            result: The result returned by process_message
            context: BackgroundContext (SSE closed, using POST updates)

        Example:
            async def on_task_complete(
                self, message: str, result: str, context: BackgroundContext
            ) -> None:
                self.logger.info(f"Background task completed for job {context.job_id}")
                await self.metrics.increment("background_tasks_completed")
        """
        pass

    async def on_task_error(
        self, message: str, error: Exception, context: BackgroundContext
    ) -> str | None:
        """
        Called when process_message raises an exception during background processing.

        This is called AFTER the SSE connection has closed, during the background
        processing phase with POST updates.

        Use for: error logging, cleanup, custom error handling

        Args:
            message: The message being processed
            error: The exception that was raised
            context: BackgroundContext (SSE closed, using POST updates)

        Returns:
            Optional custom error message to override default
            (return None to use default error message)

        Example:
            async def on_task_error(
                self, message: str, error: Exception, context: BackgroundContext
            ) -> str | None:
                self.logger.error(f"Background task failed: {error}")
                # Can still send updates via POST
                await context.update_progress("Task failed, cleaning up...", 1.0, status=TaskState.failed)

                if isinstance(error, MemoryError):
                    return "Out of memory. Try reducing batch_size parameter."
                return None  # Use default error message
        """
        return None

    async def handle_request(
        self, message: str, context: StreamingContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle async agent request: validate → ack via SSE → process with POST updates.

        For AsyncAgent, this method:
        1. Validates using updater (SSE)
        2. If rejected: Sends error via SSE and returns
        3. If accepted:
           - Sends ack via SSE (updater)
           - Creates update_client (POST)
           - Immediately calls process_message() with background context
           - All agent updates POSTed, not via SSE

        The dual-updater pattern:
        - updater (context.updater): For validation/ack via SSE
        - update_client (background_context.update_client): For processing via POST

        Args:
            message: The message to process
            context: Streaming context with updater (SSE) for validation phase
            metadata: Request metadata for validation

        Returns:
            Final result from process_message() if validation passed, None if rejected
        """
        # Step 1: Validate with updater (SSE)
        validation_result = await self.validate_message(message, metadata)

        # Step 2: Handle rejection via SSE
        if isinstance(validation_result, ValidationRejected):
            self.logger.warning(f"Message validation failed: {validation_result.reason}")
            text_part = TextPart(text=f"Validation failed: {validation_result.reason}")
            msg = Message(
                message_id=str(uuid.uuid4()), role=Role.agent, parts=[Part(root=text_part)]
            )
            await context.updater.reject(message=msg)
            return None

        # Step 3: Handle acceptance (validation_result must be ValidationAccepted at this point)
        duration_msg = ""
        if validation_result.estimated_duration_seconds:
            minutes = validation_result.estimated_duration_seconds // 60
            duration_msg = (
                f" (estimated: {minutes} min)"
                if minutes > 0
                else f" (estimated: {validation_result.estimated_duration_seconds}s)"
            )

        ack_message = (
            f"Job submitted successfully{duration_msg}. Processing will continue in the background."
        )

        self.logger.info("Validation passed, sending ack via SSE")

        # Send ack via updater (SSE)
        ack_metadata: dict[str, Any] = {}
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
        await context.updater.update_status(
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
        self.logger.info(f"Launching background task for job_id={job_id}, SSE will close after ack")

        # Get task_id and context_id for background updater
        task_id = context.request_context.task_id or str(uuid.uuid4())
        context_id = context.request_context.context_id or str(uuid.uuid4())

        asyncio.create_task(
            self._run_background_work(
                message=message,
                job_id=job_id,
                api_key=api_key,
                metadata=metadata,
                task_id=task_id,
                context_id=context_id,
                user_id=context.user_id,
                thread_id=context.thread_id,
                file_access_token=context.file_access_token,
                auth_token=context.auth_token,
                extensions=context.extensions,
            )
        )

        # Return immediately - SSE connection closes, background processing continues
        return ack_message

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
        task_id: str,
        context_id: str,
        user_id: str | None = None,
        thread_id: str | None = None,
        file_access_token: str | None = None,
        auth_token: str | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        """
        Run background work with POST updates.

        This method creates the HTTP client and async context, processes the message,
        and ensures proper cleanup. It's designed to be called with asyncio.create_task()
        so the SSE connection can close immediately after ack.

        When Inspect AI logging is enabled, this method also:
        - Creates an InspectLogger for observability
        - Passes it to the BackgroundContext
        - Logs task lifecycle events
        - Writes .eval files for inspection via `inspect view`

        Args:
            message: The message to process
            job_id: Background job ID
            api_key: API key for POSTing updates to Health Universe backend
            metadata: Request metadata
            task_id: Task ID for A2A protocol
            context_id: Context ID for A2A protocol
            user_id: Optional user ID from request
            thread_id: Optional thread ID from request
            file_access_token: Optional file access token from extensions
            auth_token: Optional JWT token for inter-agent calls
            extensions: Optional list of extension URIs from message
        """
        # Get the current event loop for sync updates from ThreadPoolExecutor
        loop = asyncio.get_event_loop()

        # Create update client (owns HTTP connection)
        base_url = os.getenv("BACKGROUND_UPDATE_URL", "https://api.healthuniverse.com")
        update_client = BackgroundUpdateClient(
            api_key=api_key,
            job_id=job_id,
            base_url=base_url,
        )

        background_context = None
        inspect_logger: "InspectLogger | None" = None

        # Create InspectLogger if enabled
        if self.inspect_logging_enabled():
            try:
                from health_universe_a2a.inspect_ai.logger import InspectLogger, set_current_logger

                inspect_logger = InspectLogger(
                    task_name=self.get_agent_name(),
                    model=self.get_model_name(),
                    log_dir=self.get_inspect_log_dir(),
                    input_description=message[:500] if message else "A2A Agent Execution",
                )

                # Set as current logger so SDK operations can find it
                set_current_logger(inspect_logger)

                self.logger.debug(f"InspectLogger created for job {job_id}")
            except ImportError:
                self.logger.warning(
                    "inspect_ai not available, Inspect logging disabled. "
                    "Install with: pip install inspect-ai"
                )
            except Exception as e:
                self.logger.warning(f"Failed to create InspectLogger: {e}")

        try:
            # Build background context with POST updater and inspect logger
            background_context = BackgroundContext(
                user_id=user_id,
                thread_id=thread_id,
                file_access_token=file_access_token,
                auth_token=auth_token,
                metadata=metadata,
                extensions=extensions,
                job_id=job_id,
                update_client=update_client,
                loop=loop,  # Enable sync updates from ThreadPoolExecutor
                inspect_logger=inspect_logger,
            )

            # Get timeout from agent configuration
            max_duration = self.get_max_duration_seconds()
            self.logger.info(
                f"Background processing started for job {job_id} with {max_duration}s timeout"
            )

            # Log task start to Inspect
            if inspect_logger:
                inspect_logger.log_task_state("working", "Starting background task")

            # Wrap processing with timeout enforcement
            async def _process_with_hooks() -> str:
                """Helper to wrap processing with lifecycle hooks."""
                await self.on_task_start(message, background_context)
                result = await self.process_message(message, background_context)
                await self.on_task_complete(message, result, background_context)
                return result

            try:
                # Use asyncio.wait_for for timeout enforcement (compatible with Python 3.7+)
                final_message = await asyncio.wait_for(_process_with_hooks(), timeout=max_duration)

            except asyncio.TimeoutError:
                # Handle timeout specifically
                timeout_msg = (
                    f"Task exceeded maximum duration of {max_duration} seconds "
                    f"({max_duration // 60} minutes)"
                )
                self.logger.error(f"Background task timed out for job {job_id}: {timeout_msg}")

                # Log timeout to Inspect
                if inspect_logger:
                    inspect_logger.log_task_state("failed", timeout_msg)

                # Call error hook for timeout
                custom_error = await self.on_task_error(
                    message, TimeoutError(timeout_msg), background_context
                )
                error_message = custom_error or timeout_msg

                # POST timeout failure
                await update_client.post_failure(error_message)
                return

            # Log success to Inspect
            if inspect_logger:
                inspect_logger.log_task_state("completed", "Task completed successfully")

            # POST completion
            self.logger.info(f"Background processing completed for job {job_id}")
            await update_client.post_completion(final_message)

        except Exception as e:
            self.logger.error(f"Background work failed for job {job_id}: {e}", exc_info=True)

            # Log error to Inspect
            if inspect_logger:
                inspect_logger.log_task_state("failed", str(e))

            # Call error hook to get custom error message
            if background_context:
                custom_error = await self.on_task_error(message, e, background_context)
            else:
                custom_error = None

            error_message = custom_error or str(e)

            # POST failure with custom or default error message
            await update_client.post_failure(error_message)

        finally:
            # Finalize InspectLogger and write .eval file
            if inspect_logger:
                try:
                    from health_universe_a2a.inspect_ai.logger import set_current_logger

                    # Clear current logger
                    set_current_logger(None)

                    # Finalize writes the .eval file
                    inspect_logger.finalize()
                    self.logger.debug(f"InspectLogger finalized for job {job_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to finalize InspectLogger: {e}")

            # Always close HTTP client to prevent connection leaks
            await update_client.close()
