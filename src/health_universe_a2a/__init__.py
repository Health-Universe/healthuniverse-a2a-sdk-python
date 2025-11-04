"""
Health Universe A2A SDK for Python

A simple, batteries-included SDK for building A2A-compliant agents.
"""

# Core agent classes
from typing import Any

from health_universe_a2a.async_agent import AsyncAgent
from health_universe_a2a.base import A2AAgent

# Context classes
from health_universe_a2a.context import AsyncContext, MessageContext

# Inter-agent communication
from health_universe_a2a.inter_agent import AgentResponse, InterAgentClient
from health_universe_a2a.streaming import StreamingAgent

# Extension types
from health_universe_a2a.types.extensions import AgentExtension

# Validation types
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)

# Server utilities (optional - requires server extra)
try:
    from health_universe_a2a.server import create_app, serve

    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

    def create_app(agent: "A2AAgent", task_store: Any | None = None) -> Any:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: pip install "health-universe-a2a[server]"'
        )

    def serve(
        agent: "A2AAgent",
        host: str | None = None,
        port: int | None = None,
        reload: bool | None = None,
        log_level: str = "info",
        task_store: Any | None = None,
    ) -> None:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: pip install "health-universe-a2a[server]"'
        )


# A2A protocol types (re-exported for convenience)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
)

__version__ = "0.1.0"

__all__ = [
    # Agent classes
    "A2AAgent",
    "StreamingAgent",
    "AsyncAgent",
    # Context classes
    "MessageContext",
    "AsyncContext",
    # Validation types
    "ValidationAccepted",
    "ValidationRejected",
    "ValidationResult",
    # Extension types
    "AgentExtension",
    # Inter-agent communication
    "InterAgentClient",
    "AgentResponse",
    # Server utilities
    "create_app",
    "serve",
    # A2A protocol types
    "AgentCard",
    "AgentProvider",
    "AgentCapabilities",
    "AgentSkill",
]
