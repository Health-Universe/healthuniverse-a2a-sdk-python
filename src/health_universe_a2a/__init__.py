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
from health_universe_a2a.inter_agent import (
    AgentRegistry,
    AgentResponse,
    InterAgentClient,
    get_agent_registry,
)

# NestJS client for S3 storage
from health_universe_a2a.nest_client import NestJSClient

# Server utilities (optional - requires server extra)
from health_universe_a2a.server import (
    create_app,
    create_multi_agent_app,
    serve,
    serve_multi_agents,
)

# Storage utilities
from health_universe_a2a.storage import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    create_storage_backend,
    directory_context,
    storage_context,
)
from health_universe_a2a.streaming import StreamingAgent

# Extension types and constants
from health_universe_a2a.types.extensions import (
    BACKGROUND_JOB_EXTENSION_URI,
    FILE_ACCESS_EXTENSION_URI,
    FILE_ACCESS_EXTENSION_URI_V2,
    HU_LOG_LEVEL_EXTENSION_URI,
    BackgroundJobExtensionParams,
    BackgroundJobExtensionResponse,
    BackgroundTaskResults,
    FileAccessExtensionContext,
    FileAccessExtensionParams,
    UpdateImportance,
    ack_background_job_enqueued,
    notify_on_task_completion,
)

# Validation types
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)

# Update client utilities
from health_universe_a2a.update_client import (
    BackgroundArtifactQueue,
    BackgroundTaskUpdater,
    BackgroundUpdateClient,
    create_background_updater,
)

__version__ = "0.2.0"

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
    # Extension types and constants
    "AgentExtension",
    "FILE_ACCESS_EXTENSION_URI",
    "FILE_ACCESS_EXTENSION_URI_V2",
    "BACKGROUND_JOB_EXTENSION_URI",
    "HU_LOG_LEVEL_EXTENSION_URI",
    "UpdateImportance",
    "FileAccessExtensionParams",
    "FileAccessExtensionContext",
    "BackgroundJobExtensionParams",
    "BackgroundJobExtensionResponse",
    "BackgroundTaskResults",
    "ack_background_job_enqueued",
    "notify_on_task_completion",
    # Inter-agent communication
    "InterAgentClient",
    "AgentResponse",
    "AgentRegistry",
    "get_agent_registry",
    # Storage utilities
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    "create_storage_backend",
    "storage_context",
    "directory_context",
    # NestJS client
    "NestJSClient",
    # Update client utilities
    "BackgroundUpdateClient",
    "BackgroundArtifactQueue",
    "BackgroundTaskUpdater",
    "create_background_updater",
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
