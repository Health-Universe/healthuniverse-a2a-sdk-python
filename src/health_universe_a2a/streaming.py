"""StreamingAgent for short-running, streaming tasks"""

import logging
import uuid
from abc import abstractmethod
from typing import Any

from a2a.types import AgentExtension, Message, Part, Role, TextPart

from health_universe_a2a.base import A2AAgentBase
from health_universe_a2a.context import StreamingContext
from health_universe_a2a.types.extensions import FILE_ACCESS_EXTENSION_URI
from health_universe_a2a.types.validation import ValidationRejected

logger = logging.getLogger(__name__)


class StreamingAgent(A2AAgentBase):
    """
    Agent for short-running tasks (< 5 min) with SSE streaming updates.

    **When to Use StreamingAgent:**
    ✓ Task completes in seconds to a few minutes (< 5 min)
    ✓ Users expect immediate, real-time feedback
    ✓ You want to stream progress updates live via SSE
    ✓ Task duration fits within HTTP/SSE timeout limits

    **Use AsyncAgent instead if:**
    ✗ Task takes more than 5 minutes
    ✗ Task might run for hours
    ✗ SSE connection timeouts are a concern

    The agent automatically:
    - Enables streaming in the agent card
    - Configures file access extension if requires_file_access() returns True
    - Streams progress updates via A2A protocol
    - Handles task lifecycle (validation, processing, completion)

    Example:
        from health_universe_a2a import StreamingAgent, StreamingContext
        import json

        class QuickAnalysisAgent(StreamingAgent):
            def get_agent_name(self) -> str:
                return "Quick Analyzer"

            def get_agent_description(self) -> str:
                return "Performs quick data analysis"

            async def process_message(self, message: str, context: StreamingContext) -> str:
                await context.update_progress("Loading data...", 0.2)
                data = await self.load_data(message)

                await context.update_progress("Analyzing...", 0.6)
                results = await self.analyze(data)

                await context.add_artifact(
                    name="Analysis Results",
                    content=json.dumps(results),
                    data_type="application/json"
                )

                return "Analysis complete!"

    Implementation Details:
        - Inherits from A2AAgentBase and implements AgentExecutor interface
        - Integrates with a2a-sdk's DefaultRequestHandler and TaskUpdater
        - Updates streamed in real-time via SSE (Server-Sent Events)
        - Connection remains open throughout processing
        - Completion sent via TaskUpdater.complete() when done
        - Best for tasks completing within typical HTTP/SSE timeout (~5 minutes)
    """

    def supports_streaming(self) -> bool:
        """Enable streaming support."""
        return True

    def get_extensions(self) -> list[AgentExtension]:
        """Auto-configure extensions based on requirements."""
        extensions: list[AgentExtension] = []

        # Add file access if needed
        if self.requires_file_access():
            extensions.append(AgentExtension(uri=FILE_ACCESS_EXTENSION_URI))

        return extensions

    @abstractmethod
    async def process_message(self, message: str, context: StreamingContext) -> str:
        """
        Process a message with streaming context.

        This method is called after validation passes. All updates are sent
        via SSE (Server-Sent Events) to the client in real-time.

        Args:
            message: The message to process
            context: StreamingContext with updater for sending SSE updates

        Returns:
            Final result message

        Example:
            async def process_message(self, message: str, context: StreamingContext) -> str:
                await context.update_progress("Starting...", 0.1)
                result = await self.do_work(message)
                await context.add_artifact("Results", json.dumps(result), "application/json")
                return "Processing complete!"
        """
        pass

    async def on_task_start(self, message: str, context: StreamingContext) -> None:  # noqa: B027
        """
        Called after validation passes, before process_message.

        Use for: logging, metrics, setup

        Args:
            message: The message being processed
            context: StreamingContext (SSE is open)

        Example:
            async def on_task_start(self, message: str, context: StreamingContext) -> None:
                self.logger.info(f"Starting task for user {context.user_id}")
                await self.metrics.increment("tasks_started")
        """
        pass

    async def on_task_complete(self, message: str, result: str, context: StreamingContext) -> None:  # noqa: B027
        """
        Called after process_message completes successfully.

        Use for: logging, metrics, cleanup

        Args:
            message: The message that was processed
            result: The result returned by process_message
            context: StreamingContext (SSE is still open)

        Example:
            async def on_task_complete(
                self, message: str, result: str, context: StreamingContext
            ) -> None:
                self.logger.info(f"Task completed for user {context.user_id}")
                await self.metrics.increment("tasks_completed")
        """
        pass

    async def on_task_error(
        self, message: str, error: Exception, context: StreamingContext
    ) -> str | None:
        """
        Called when process_message raises an exception during SSE streaming.

        Use for: error logging, cleanup, custom error handling

        Args:
            message: The message being processed
            error: The exception that was raised
            context: StreamingContext (SSE is still open)

        Returns:
            Optional custom error message to override default
            (return None to use default error message)

        Example:
            async def on_task_error(
                self, message: str, error: Exception, context: StreamingContext
            ) -> str | None:
                self.logger.error(f"Task failed: {error}")
                if isinstance(error, TimeoutError):
                    return "Processing timed out. Please try a smaller request."
                return None  # Use default error message
        """
        return None

    async def handle_request(
        self, message: str, context: StreamingContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle streaming agent request: validate → process → return result.

        For StreamingAgent, this method:
        1. Validates the message
        2. If rejected: Sends error via SSE and returns None
        3. If accepted: Calls process_message() and returns result
        4. The base class execute() sends the final completion via SSE

        Args:
            message: The message to process
            context: Streaming context with updater (SSE)
            metadata: Request metadata for validation

        Returns:
            Final result from process_message() if validation passed, None if rejected
        """
        # Step 1: Validate
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

        # Step 3: Call lifecycle hook
        await self.on_task_start(message, context)

        try:
            # Step 4: Process message
            result = await self.process_message(message, context)

            # Step 5: Call lifecycle hook
            await self.on_task_complete(message, result, context)

            return result

        except Exception as error:
            # Call error hook and optionally get custom error message
            custom_error = await self.on_task_error(message, error, context)

            # Send error update with custom message if provided
            error_message = custom_error or str(error)
            await context.update_progress(error_message, status="error")

            # Always re-raise the original error
            raise
