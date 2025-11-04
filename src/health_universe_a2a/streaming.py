"""StreamingAgent for short-running, streaming tasks"""

import logging

from health_universe_a2a.base import A2AAgent
from health_universe_a2a.context import MessageContext
from health_universe_a2a.types.extensions import FILE_ACCESS_EXTENSION_URI, AgentExtension

logger = logging.getLogger(__name__)


class StreamingAgent(A2AAgent[MessageContext]):
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

    Implementation Details:
        - Inherits from A2AAgent and implements AgentExecutor interface
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
        extensions = []

        # Add file access if needed
        if self.requires_file_access():
            extensions.append(AgentExtension(uri=FILE_ACCESS_EXTENSION_URI))

        return extensions
