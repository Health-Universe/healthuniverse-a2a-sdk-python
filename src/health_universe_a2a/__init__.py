"""
Health Universe A2A SDK for Python

A simple, batteries-included SDK for building A2A-compliant agents.
"""

# A2A protocol types (re-exported for convenience)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentProvider,
    AgentSkill,
)

from health_universe_a2a.async_agent import AsyncAgent
from health_universe_a2a.base import A2AAgentBase

# Context classes
from health_universe_a2a.context import BackgroundContext, BaseContext, StreamingContext

# Inter-agent communication
from health_universe_a2a.inter_agent import AgentResponse, InterAgentClient

# Server utilities (optional - requires server extra)
from health_universe_a2a.server import (
    create_app,
    create_multi_agent_app,
    serve,
    serve_multi_agents,
)
from health_universe_a2a.streaming import StreamingAgent

# Validation types
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)

__version__ = "0.1.0"

__all__ = [
    # Agent classes
    "A2AAgentBase",
    "StreamingAgent",
    "AsyncAgent",
    # Context classes
    "BaseContext",
    "StreamingContext",
    "BackgroundContext",
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
    "create_multi_agent_app",
    "serve",
    "serve_multi_agents",
    # A2A protocol types
    "AgentCard",
    "AgentProvider",
    "AgentCapabilities",
    "AgentSkill",
]
