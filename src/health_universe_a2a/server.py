"""Server utilities for running A2A agents with HTTP endpoints."""

import logging
import os
from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from health_universe_a2a.base import A2AAgentBase

logger = logging.getLogger(__name__)


def create_app(agent: A2AAgentBase, task_store: Any | None = None) -> Any:
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
        from health_universe_a2a import StreamingAgent, StreamingContext, create_app
        import uvicorn

        class MyAgent(StreamingAgent):
            def get_agent_name(self) -> str:
                return "My Agent"

            def get_agent_description(self) -> str:
                return "Does something useful"

            async def process_message(self, message: str, context: StreamingContext) -> str:
                return f"Processed: {message}"

        agent = MyAgent()
        app = create_app(agent)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """

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
    agent: A2AAgentBase,
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
        agent: The A2AAgentBase instance to serve (StreamingAgent or AsyncAgent)
        host: Server host (overrides env var)
        port: Server port (overrides env var)
        reload: Enable auto-reload (overrides env var)
        log_level: Uvicorn log level (default: "info")
        task_store: Optional task store (defaults to InMemoryTaskStore)

    Raises:
        ImportError: If uvicorn or a2a server dependencies are not installed

    Example:
        from health_universe_a2a import StreamingAgent, StreamingContext, serve

        class MyAgent(StreamingAgent):
            def get_agent_name(self) -> str:
                return "My Agent"

            def get_agent_description(self) -> str:
                return "Does something useful"

            async def process_message(self, message: str, context: StreamingContext) -> str:
                return f"Processed: {message}"

        if __name__ == "__main__":
            agent = MyAgent()
            serve(agent)  # Starts server on http://0.0.0.0:8000
    """

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


def create_multi_agent_app(
    agents: dict[str, A2AAgentBase],
    task_store: Any | None = None,
) -> Any:
    """
    Create a Starlette ASGI application with multiple A2A agents mounted at different paths.

    This enables multi-agent architectures where agents can call each other
    using relative paths (e.g., "/document-reader") within the same server.

    Args:
        agents: Dictionary mapping mount paths to agent instances.
                Paths must start with "/" (e.g., {"/orchestrator": orchestrator})
        task_store: Optional shared task store for all agents (defaults to InMemoryTaskStore)

    Returns:
        Starlette application instance with all agents mounted

    Raises:
        ValueError: If any path doesn't start with "/"
        ImportError: If a2a server dependencies are not installed

    Example:
        from health_universe_a2a import create_multi_agent_app
        import uvicorn

        orchestrator = OrchestratorAgent()
        reader = DocumentReaderAgent()

        app = create_multi_agent_app({
            "/orchestrator": orchestrator,
            "/document-reader": reader,
        })

        # Full control over uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8501, workers=4)

        # Or mount into a larger Starlette app
        from starlette.applications import Starlette
        from starlette.routing import Mount

        main_app = Starlette(routes=[
            Mount("/api", app=your_api_app),
            Mount("/agents", app=app),  # Multi-agent app as submount
        ])
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount

    # Validate paths
    for path in agents.keys():
        if not path.startswith("/"):
            raise ValueError(f"Agent path must start with '/': {path}")

    logger.info(f"Creating multi-agent app with {len(agents)} agent(s)")

    # Create routes for each agent
    routes: list[Mount] = []
    for path, agent in agents.items():
        logger.info(f"  Mounting {agent.get_agent_name()} at {path}")
        agent_app = create_app(agent, task_store=task_store)
        routes.append(Mount(path, app=agent_app))

    # Build and return main app
    main_app = Starlette(routes=routes)
    return main_app


def serve_multi_agents(
    agents: dict[str, A2AAgentBase],
    host: str | None = None,
    port: int | None = None,
    reload: bool | None = None,
    log_level: str = "info",
    task_store: Any | None = None,
) -> None:
    """
    Start an HTTP server hosting multiple A2A agents at different paths.

    This is a convenience method that creates the app using create_multi_agent_app()
    and runs uvicorn. For more control, use create_multi_agent_app() directly.

    Environment variables:
        HOST: Server host (default: "0.0.0.0")
        PORT or AGENT_PORT: Server port (default: 8501 for multi-agent setups)
        RELOAD: Enable auto-reload on code changes (default: "false")

    Args:
        agents: Dictionary mapping mount paths to agent instances.
                Paths must start with "/" (e.g., {"/orchestrator": orchestrator})
        host: Server host (overrides env var)
        port: Server port (overrides env var, defaults to 8501)
        reload: Enable auto-reload (overrides env var)
        log_level: Uvicorn log level (default: "info")
        task_store: Optional shared task store for all agents (defaults to InMemoryTaskStore)

    Raises:
        ValueError: If any path doesn't start with "/"
        ImportError: If uvicorn or a2a server dependencies are not installed

    Example:
        from health_universe_a2a import StreamingAgent, StreamingContext, serve_multi_agents

        class OrchestratorAgent(StreamingAgent):
            def __init__(self):
                super().__init__()
                self.reader_agent = "/document-reader"  # Relative path

            def get_agent_name(self) -> str:
                return "Orchestrator"

            def get_agent_description(self) -> str:
                return "Coordinates document processing"

            async def process_message(self, message: str, context: StreamingContext) -> str:
                # Call other agent using relative path
                response = await self.call_other_agent(
                    self.reader_agent,
                    message,
                    context
                )
                return f"Orchestrated: {response.text}"

        class DocumentReaderAgent(StreamingAgent):
            def get_agent_name(self) -> str:
                return "Document Reader"

            def get_agent_description(self) -> str:
                return "Reads documents"

            async def process_message(self, message: str, context: StreamingContext) -> str:
                return f"Read document: {message}"

        if __name__ == "__main__":
            orchestrator = OrchestratorAgent()
            reader = DocumentReaderAgent()

            serve_multi_agents({
                "/orchestrator": orchestrator,
                "/document-reader": reader,
            }, port=8501)

            # Now agents can be called:
            # - http://localhost:8501/orchestrator/
            # - http://localhost:8501/document-reader/
    """

    # Configuration from environment with overrides
    # Default port is 8501 for multi-agent setups (avoiding conflict with single-agent 8000)
    actual_host = host if host is not None else os.getenv("HOST", "0.0.0.0")
    actual_port = (
        port if port is not None else int(os.getenv("PORT", os.getenv("AGENT_PORT", "8501")))
    )
    actual_reload: bool = (
        reload if reload is not None else os.getenv("RELOAD", "false").lower() == "true"
    )

    # Create the multi-agent app
    app = create_multi_agent_app(agents, task_store=task_store)

    # Log startup information
    logger.info("=" * 60)
    logger.info(f"🚀 Starting multi-agent server on http://{actual_host}:{actual_port}")
    logger.info("=" * 60)
    logger.info(f"📋 Mounted {len(agents)} agent(s):")
    for path, agent in agents.items():
        logger.info(f"   {path:25} → {agent.get_agent_name()} v{agent.get_agent_version()}")
        logger.info(
            f"      Agent Card: http://localhost:{actual_port}{path}/.well-known/agent-card.json"
        )
        logger.info(
            f'      JSON-RPC:   POST http://localhost:{actual_port}{path}/ (method: "message/send")'
        )
    logger.info("=" * 60)
    logger.info("💡 Agents can call each other using relative paths:")
    logger.info(
        f"   Example: await self.call_other_agent('{list(agents.keys())[0]}', message, context)"
    )
    logger.info("=" * 60)

    # Run the server
    uvicorn.run(
        app,
        host=actual_host,
        port=actual_port,
        reload=actual_reload,
        log_level=log_level,
    )
