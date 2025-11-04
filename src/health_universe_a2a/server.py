"""Server utilities for running A2A agents with HTTP endpoints."""

import logging
import os
from typing import Any

try:
    import uvicorn
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

from health_universe_a2a.base import A2AAgent

logger = logging.getLogger(__name__)


def create_app(agent: A2AAgent, task_store: Any | None = None) -> Any:
    """
    Create a Starlette ASGI application for an A2A agent.

    This creates a production-ready HTTP server with all required A2A endpoints:
    - Agent card endpoint: /.well-known/agent-card.json
    - JSON-RPC endpoint: POST / (method: "message/send")
    - Health check: /health

    Args:
        agent: The A2AAgent instance to serve
        task_store: Optional task store (defaults to InMemoryTaskStore)

    Returns:
        Starlette application instance

    Raises:
        ImportError: If uvicorn or a2a server dependencies are not installed

    Example:
        from health_universe_a2a import A2AAgent, create_app
        import uvicorn

        class MyAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "My Agent"

            def get_agent_description(self) -> str:
                return "Does something useful"

            async def process_message(self, message: str, context) -> str:
                return f"Processed: {message}"

        agent = MyAgent()
        app = create_app(agent)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    if not UVICORN_AVAILABLE:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: uv pip install "health-universe-a2a[server]" or pip install uvicorn a2a'
        )

    logger.info(f"Creating app for {agent.get_agent_name()} v{agent.get_agent_version()}")

    # Create agent card
    agent_card = agent.create_agent_card()

    # Create task store if not provided
    if task_store is None:
        task_store = InMemoryTaskStore()

    # A2AAgent now directly implements AgentExecutor interface
    # Pass agent directly to DefaultRequestHandler
    request_handler = DefaultRequestHandler(agent_executor=agent, task_store=task_store)

    # Build Starlette app with A2A endpoints
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler).build()

    return app


def serve(
    agent: A2AAgent,
    host: str | None = None,
    port: int | None = None,
    reload: bool | None = None,
    log_level: str = "info",
    task_store: Any | None = None,
) -> None:
    """
    Start an HTTP server for an A2A agent.

    This is a convenience method that creates the app and runs uvicorn.
    Configuration is read from environment variables with sensible defaults.

    Environment variables:
        HOST: Server host (default: "0.0.0.0")
        PORT or AGENT_PORT: Server port (default: 8000)
        RELOAD: Enable auto-reload on code changes (default: "false")

    Args:
        agent: The A2AAgent instance to serve
        host: Server host (overrides env var)
        port: Server port (overrides env var)
        reload: Enable auto-reload (overrides env var)
        log_level: Uvicorn log level (default: "info")
        task_store: Optional task store (defaults to InMemoryTaskStore)

    Raises:
        ImportError: If uvicorn or a2a server dependencies are not installed

    Example:
        from health_universe_a2a import A2AAgent, serve

        class MyAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "My Agent"

            def get_agent_description(self) -> str:
                return "Does something useful"

            async def process_message(self, message: str, context) -> str:
                return f"Processed: {message}"

        if __name__ == "__main__":
            agent = MyAgent()
            serve(agent)  # Starts server on http://0.0.0.0:8000
    """
    if not UVICORN_AVAILABLE:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: uv pip install "health-universe-a2a[server]" or pip install uvicorn a2a'
        )

    # Configuration from environment with overrides
    actual_host = host if host is not None else os.getenv("HOST", "0.0.0.0")
    actual_port = (
        port if port is not None else int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    )
    actual_reload: bool = (
        reload if reload is not None else os.getenv("RELOAD", "false").lower() == "true"
    )

    # Create the app
    app = create_app(agent, task_store=task_store)

    # Log startup information
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {agent.get_agent_name()} on http://{actual_host}:{actual_port}")
    logger.info("=" * 60)
    logger.info("📋 Endpoints:")
    logger.info(f"   Agent Card: http://localhost:{actual_port}/.well-known/agent-card.json")
    logger.info(f'   JSON-RPC:   POST http://localhost:{actual_port}/ (method: "message/send")')
    logger.info(f"   Health:     http://localhost:{actual_port}/health")
    logger.info("=" * 60)
    logger.info(f"📦 Agent: {agent.get_agent_name()} v{agent.get_agent_version()}")
    logger.info(f"📝 Description: {agent.get_agent_description()}")
    logger.info("=" * 60)

    # Run the server
    uvicorn.run(
        app,
        host=actual_host,
        port=actual_port,
        reload=actual_reload,
        log_level=log_level,
    )
