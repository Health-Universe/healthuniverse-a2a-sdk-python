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
from health_universe_a2a.types.extensions import NavigatorTaskStatus, UpdateImportance
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
)

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
    "NavigatorTaskStatus",
    "create_app",
    "serve_multi_agents",
    # Aliases (for explicit imports)
    "AsyncAgent",
    "A2AAgentBase",
    "BackgroundContext",
    "BaseContext",
    "create_multi_agent_app",
]
