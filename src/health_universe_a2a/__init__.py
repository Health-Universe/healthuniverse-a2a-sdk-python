"""
Health Universe A2A SDK for Python

A batteries-included SDK for building A2A-compliant agents on the Health Universe platform.

Quick Start:
    from health_universe_a2a import Agent, AgentContext

    class MyAgent(Agent):
        def get_agent_name(self) -> str:
            return "My Agent"

        def get_agent_description(self) -> str:
            return "Does something useful"

        async def process_message(self, message: str, context: AgentContext) -> str:
            return f"Processed: {message}"

    if __name__ == "__main__":
        agent = MyAgent()
        agent.serve()

Inspect AI Integration:
    The SDK includes built-in Inspect AI integration for observability.
    All agent executions automatically generate .eval log files viewable via `inspect view`.

    # View logs after running your agent
    $ inspect view --log-dir ./inspect_logs

    # Manual logging in your agent
    async def process_message(self, message: str, context: AgentContext) -> str:
        # SDK operations are automatically logged
        docs = await context.document_client.list_documents()

        # Manual logging for custom operations
        if context.inspect_logger:
            context.inspect_logger.log_tool_call("my_tool", {...}, result, duration)

        return "Done!"

    Configuration:
        INSPECT_LOG_DIR: Directory for .eval files (default: ./inspect_logs)
        INSPECT_LOGGING_ENABLED: Enable/disable logging (default: true)
        INSPECT_EVAL_MODE: Run through inspect_eval (default: true)
"""

from health_universe_a2a.async_agent import AsyncAgent
from health_universe_a2a.base import A2AAgentBase
from health_universe_a2a.context import BackgroundContext, BaseContext
from health_universe_a2a.documents import Document, DocumentClient
from health_universe_a2a.server import (
    create_app,
    create_multi_agent_app,
    serve_multi_agents,
)
from health_universe_a2a.types.extensions import UpdateImportance
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
)

# Inspect AI integration (lazy imports to avoid hard dependency)
try:
    from health_universe_a2a.inspect_ai import (
        InspectLogger,
        InspectLoggerContext,
        InspectLoggerAsyncContext,
        create_agent_task,
        run_agent_as_eval,
        start_inspect_view,
        stop_inspect_view,
        is_viewer_running,
        get_viewer_url,
        get_viewer_status,
        inspect_logging_enabled,
        inspect_eval_enabled,
    )

    _INSPECT_AVAILABLE = True
except ImportError:
    # inspect_ai not installed - provide stubs for type checking
    InspectLogger = None  # type: ignore[misc, assignment]
    InspectLoggerContext = None  # type: ignore[misc, assignment]
    InspectLoggerAsyncContext = None  # type: ignore[misc, assignment]
    create_agent_task = None  # type: ignore[misc, assignment]
    run_agent_as_eval = None  # type: ignore[misc, assignment]
    start_inspect_view = None  # type: ignore[misc, assignment]
    stop_inspect_view = None  # type: ignore[misc, assignment]
    is_viewer_running = None  # type: ignore[misc, assignment]
    get_viewer_url = None  # type: ignore[misc, assignment]
    get_viewer_status = None  # type: ignore[misc, assignment]
    inspect_logging_enabled = None  # type: ignore[misc, assignment]
    inspect_eval_enabled = None  # type: ignore[misc, assignment]
    _INSPECT_AVAILABLE = False

# Recommended aliases for simpler API
Agent = AsyncAgent
AgentContext = BackgroundContext

__version__ = "0.2.0"

__all__ = [
    # Core (what 95% of users need)
    "Agent",
    "AgentContext",
    "Document",
    "DocumentClient",
    "ValidationAccepted",
    "ValidationRejected",
    "UpdateImportance",
    "create_app",
    "serve_multi_agents",
    # Aliases (for explicit imports)
    "AsyncAgent",
    "A2AAgentBase",
    "BackgroundContext",
    "BaseContext",
    "create_multi_agent_app",
    # Inspect AI integration
    "InspectLogger",
    "InspectLoggerContext",
    "InspectLoggerAsyncContext",
    "create_agent_task",
    "run_agent_as_eval",
    "start_inspect_view",
    "stop_inspect_view",
    "is_viewer_running",
    "get_viewer_url",
    "get_viewer_status",
    "inspect_logging_enabled",
    "inspect_eval_enabled",
]
