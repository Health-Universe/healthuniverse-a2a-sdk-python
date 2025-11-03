"""StreamingAgent for short-running, streaming tasks"""

import logging
from typing import Any

from health_universe_a2a.base import A2AAgent
from health_universe_a2a.context import MessageContext
from health_universe_a2a.types.extensions import FILE_ACCESS_EXTENSION_URI, AgentExtension

logger = logging.getLogger(__name__)


class StreamingAgent(A2AAgent):
    """
    Agent for short-running tasks (< 5 min) with SSE streaming updates.

    Use this agent type when:
    - Users expect immediate feedback
    - Processing takes seconds to a few minutes
    - You want to stream progress updates via A2A protocol (SSE)
    - Task can complete within typical HTTP/SSE timeout limits

    The agent automatically:
    - Enables streaming in the agent card
    - Configures file access extension if requires_file_access() returns True
    - Streams progress updates via A2A protocol
    - Handles task lifecycle (validation, processing, completion)

    Example:
        from health_universe_a2a import StreamingAgent, MessageContext

        class QuickAnalysisAgent(StreamingAgent):
            def get_agent_name(self) -> str:
                return "Quick Analyzer"

            def get_agent_description(self) -> str:
                return "Performs quick data analysis"

            async def process_message(self, message: str, context: MessageContext) -> str:
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

    Note:
        This is a stub implementation. The actual integration with the A2A SDK
        (AgentExecutor, EventQueue, etc.) should be implemented based on your
        specific A2A framework.
    """

    def supports_streaming(self) -> bool:
        """Enable streaming support."""
        return True

    def get_extensions(self) -> list[AgentExtension]:
        """Auto-configure extensions based on requirements."""
        extensions = []

        # Add file access if needed
        if self.requires_file_access():
            extensions.append(AgentExtension(uri=FILE_ACCESS_EXTENSION_URI))

        return extensions

    # The actual execute() method would integrate with your A2A SDK here
    # This would handle:
    # 1. Extracting message from RequestContext
    # 2. Running validation
    # 3. Creating TaskUpdater for streaming
    # 4. Calling process_message()
    # 5. Streaming updates via EventQueue
    # 6. Handling errors and completion

    def _build_message_context(
        self, request_context: Any, updater: Any
    ) -> MessageContext:
        """
        Build simplified context from A2A request context.

        Args:
            request_context: A2A RequestContext
            updater: TaskUpdater for streaming updates

        Returns:
            MessageContext for agent use
        """
        # This would extract metadata from your A2A SDK's RequestContext
        # Placeholder implementation:
        metadata: dict[str, Any] = {}
        file_access_token = None

        # Extract file access token if present
        if FILE_ACCESS_EXTENSION_URI in metadata:
            file_access_token = metadata[FILE_ACCESS_EXTENSION_URI].get("access_token")

        return MessageContext(
            user_id=metadata.get("user_id"),
            thread_id=metadata.get("thread_id"),
            file_access_token=file_access_token,
            metadata=metadata,
            _updater=updater,
            _request_context=request_context,
        )
