"""Base A2A Agent class for Health Universe agents.

This module provides the A2AAgentBase class that all Health Universe agents inherit from.
For most use cases, use the Agent class (alias for AsyncAgent) from the main package.

Example:
    from health_universe_a2a import Agent, AgentContext

    class MyAgent(Agent):
        def get_agent_name(self) -> str:
            return "My Agent"

        def get_agent_description(self) -> str:
            return "Processes medical data"

        async def process_message(self, message: str, context: AgentContext) -> str:
            await context.update_progress("Processing...", 0.5)
            return f"Processed: {message}"

    if __name__ == "__main__":
        agent = MyAgent()
        agent.serve()
"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
    TextPart,
)

from health_universe_a2a.context import BaseContext, StreamingContext
from health_universe_a2a.inter_agent import AgentResponse, InterAgentClient
from health_universe_a2a.types.extensions import FILE_ACCESS_EXTENSION_URI
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class A2AAgentBase(AgentExecutor, ABC):
    """
    Base class providing shared configuration and utilities for A2A agents.

    This class contains all shared functionality like agent card creation,
    validation, inter-agent communication, and lifecycle hooks.

    For most use cases, use Agent (alias for AsyncAgent) instead of this class directly.

    Subclasses must implement:
        - get_agent_name(): Return the agent's display name
        - get_agent_description(): Return a description of what the agent does
        - process_message(message, context): Process messages and return results

    Optional overrides:
        - validate_message(): Validate incoming messages before processing
        - get_agent_version(): Custom version string
        - get_supported_input_formats(): Supported input MIME types
        - get_supported_output_formats(): Supported output MIME types
        - Lifecycle hooks: on_startup(), on_shutdown(), etc.

    Example:
        from health_universe_a2a import Agent, AgentContext

        class DataAnalyzer(Agent):
            def get_agent_name(self) -> str:
                return "Data Analyzer"

            def get_agent_description(self) -> str:
                return "Analyzes clinical datasets"

            async def process_message(self, message: str, context: AgentContext) -> str:
                # List documents in the thread
                docs = await context.documents.list_documents()

                # Process with progress updates
                await context.update_progress("Analyzing...", 0.5)

                return "Analysis complete!"
    """

    def __init__(self) -> None:
        """Initialize the agent with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)

    # Required abstract methods for configuration

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the agent's name for the AgentCard.

        Returns:
            Agent name (e.g., "Data Analyzer", "Protocol Generator")

        Example:
            def get_agent_name(self) -> str:
                return "Clinical Data Analyzer"
        """
        pass

    @abstractmethod
    def get_agent_description(self) -> str:
        """
        Return the agent's description for the AgentCard.

        Returns:
            Agent description (e.g., "Analyzes datasets and returns insights")

        Example:
            def get_agent_description(self) -> str:
                return "Analyzes clinical trial data and generates statistical reports"
        """
        pass

    @abstractmethod
    async def handle_request(
        self, message: str, context: StreamingContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle an incoming request (internal method).

        This is called by execute() and must be implemented by subclasses.
        For AsyncAgent, this handles validation via SSE, then spawns background processing.

        Args:
            message: The extracted message text
            context: Internal SSE context for validation phase
            metadata: Request metadata

        Returns:
            Result string or None if validation rejected
        """
        pass

    # AgentExecutor interface implementation

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute agent processing for a request.

        This implements the AgentExecutor interface required by a2a-sdk's DefaultRequestHandler.

        Args:
            context: Request context from a2a-sdk
            event_queue: Event queue for sending updates

        Raises:
            Exception: Any exception from agent processing is propagated
        """
        logger.info(f"Executing agent={self.get_agent_name()} task_id={context.task_id}")

        # Ensure task_id and context_id are set
        if not context.task_id or not context.context_id:
            raise ValueError("task_id and context_id must be set in RequestContext")

        # Ensure message is present
        if not context.message:
            raise ValueError("message must be set in RequestContext")

        # Create TaskUpdater to send events
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )

        try:
            # Extract message text
            message_text = self._extract_message_text(context.message)
            logger.debug(f"Extracted message text: {message_text[:100]}...")

            # Extract user context
            user_id = None
            thread_id = None
            if context.call_context and context.call_context.user:
                user_id = getattr(context.call_context.user, "user_name", None)

            # Extract metadata from context
            metadata = context.metadata or {}

            # If metadata is empty, try to extract from message.metadata
            if not metadata and context.message:
                message_metadata = getattr(context.message, "metadata", None)
                if message_metadata and isinstance(message_metadata, dict):
                    metadata = message_metadata
                    logger.debug(
                        f"Extracted metadata from message.metadata: {list(metadata.keys())}"
                    )

            # Extract thread_id from metadata
            if "thread_id" in metadata:
                thread_id = metadata["thread_id"]

            # Extract file access token from extension
            file_access_token = None
            file_ext_data = metadata.get(FILE_ACCESS_EXTENSION_URI)

            if file_ext_data and isinstance(file_ext_data, dict):
                file_access_token = file_ext_data.get("access_token")

                # Extract user_id and thread_id from extension context if present
                if "context" in file_ext_data:
                    ext_context = file_ext_data["context"]
                    if isinstance(ext_context, dict):
                        if not user_id and "user_id" in ext_context:
                            user_id = ext_context["user_id"]
                            logger.debug(f"Extracted user_id from file_access extension: {user_id}")
                        if not thread_id and "thread_id" in ext_context:
                            thread_id = ext_context["thread_id"]
                            logger.debug(
                                f"Extracted thread_id from file_access extension: {thread_id}"
                            )

            # Extract extensions list from message
            extensions = None
            if context.message and hasattr(context.message, "extensions"):
                extensions = context.message.extensions

            # Extract JWT auth token for inter-agent calls
            auth_token = None
            if context.call_context and context.call_context.state:
                auth_token = context.call_context.state.get("authorization")
                if not auth_token:
                    auth_header = context.call_context.state.get("Authorization")
                    if auth_header and isinstance(auth_header, str):
                        auth_token = auth_header.replace("Bearer ", "").strip()

            # Build internal SSE context for validation phase
            message_context = StreamingContext(
                user_id=user_id,
                thread_id=thread_id,
                file_access_token=file_access_token,
                auth_token=auth_token,
                metadata=metadata,
                extensions=extensions,
                updater=updater,
                request_context=context,
            )

            # Call agent's handle_request
            logger.info("Calling agent.handle_request()")
            result = await self.handle_request(
                message=message_text, context=message_context, metadata=metadata
            )

            logger.info(f"Agent returned result: {result is not None}")

            # For AsyncAgent: ack already sent with final=True, background task running
            # Check if this is background mode by checking push_notifications support
            is_background_mode = self.supports_push_notifications()

            if result and not is_background_mode:
                # Not in background mode - send final message with result
                logger.info("Sending final completion message")
                text_part = TextPart(text=result)
                final_message = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(root=text_part)],
                )
                await updater.complete(message=final_message)
            elif result and is_background_mode:
                # AsyncAgent: ack already sent via SSE, background processing running with POST
                logger.info(
                    "AsyncAgent: ack sent via SSE, background processing running with POST updates"
                )

        except Exception as error:
            logger.error(f"Agent execution failed: {error}", exc_info=True)
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel the agent execution.

        **IMPORTANT: Cancellation is not fully implemented in this SDK.**

        For production use, consider:
        1. Override this method in your agent subclass
        2. Store cancellation state (e.g., in a shared dict keyed by task_id)
        3. Check that state periodically in your process_message() implementation

        Args:
            context: Request context for the task to cancel
            event_queue: Event queue for sending cancellation updates

        Example:
            # In your agent class:
            def __init__(self):
                super().__init__()
                self._cancelled_tasks = set()

            async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
                self._cancelled_tasks.add(context.task_id)
                self.logger.info(f"Marked task {context.task_id} as cancelled")

            async def process_message(self, message: str, context: AgentContext) -> str:
                if context.request_context.task_id in self._cancelled_tasks:
                    return "Task was cancelled"
                # ... rest of processing
        """
        logger.info(f"Cancelling execution for {self.get_agent_name()} task_id={context.task_id}")

    def _extract_message_text(self, message: Message) -> str:
        """Extract text content from an A2A Message."""
        text_parts: list[str] = []

        for part in message.parts:
            # Check the root discriminated union first
            if part.root is not None and isinstance(part.root, TextPart):
                text_parts.append(part.root.text)
            # Fallback for direct TextPart
            elif isinstance(part, TextPart):
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else ""

    # Optional validation hook

    async def validate_message(self, message: str, metadata: dict[str, Any]) -> ValidationResult:
        """
        Validate incoming message before processing.

        This runs BEFORE the task is enqueued. Use this to:
        - Check message format
        - Validate required parameters
        - Verify file attachments
        - Estimate processing time
        - Check quotas/permissions

        Args:
            message: The extracted message text
            metadata: Raw metadata from request (extensions, user info, etc)

        Returns:
            ValidationResult - either ValidationAccepted or ValidationRejected

        Default implementation accepts all messages.

        Example:
            from health_universe_a2a import ValidationAccepted, ValidationRejected

            async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
                if len(message) < 10:
                    return ValidationRejected(reason="Message too short (min 10 chars)")
                return ValidationAccepted(estimated_duration_seconds=60)
        """
        return ValidationAccepted()

    # Optional configuration methods

    def get_agent_version(self) -> str:
        """Return agent version. Default: "1.0.0" """
        return "1.0.0"

    def get_supported_input_formats(self) -> list[str]:
        """Supported input MIME types. Default: ["text/plain", "application/json"]"""
        return ["text/plain", "application/json"]

    def get_supported_output_formats(self) -> list[str]:
        """Supported output MIME types. Default: ["text/plain", "application/json"]"""
        return ["text/plain", "application/json"]

    def get_extensions(self) -> list[AgentExtension]:
        """
        Return list of extensions supported by this agent.

        This is automatically configured by AsyncAgent.
        Only override if you need custom extensions.
        """
        return []

    # AgentCard creation

    def create_agent_card(self) -> AgentCard:
        """
        Create the AgentCard for agent discovery.
        Fully compliant with A2A specification v0.3.0.
        """
        base_url = self.get_base_url()

        return AgentCard(
            protocol_version="0.3.0",
            name=self.get_agent_name(),
            description=self.get_agent_description(),
            version=self.get_agent_version(),
            url=base_url,
            preferred_transport="JSONRPC",
            additional_interfaces=[AgentInterface(url=base_url, transport="JSONRPC")],
            provider=AgentProvider(
                organization=self.get_provider_organization(), url=self.get_provider_url()
            ),
            capabilities=AgentCapabilities(
                streaming=self.supports_streaming(),
                push_notifications=self.supports_push_notifications(),
                extensions=self.get_extensions(),
            ),
            skills=self.get_agent_skills(),
            security_schemes=self.get_security_schemes(),
            security=self.get_security_requirements(),
            default_input_modes=self.get_supported_input_formats(),
            default_output_modes=self.get_supported_output_formats(),
        )

    def get_base_url(self) -> str:
        """Get the agent's base URL from environment."""
        return os.getenv("HU_APP_URL", os.getenv("A2A_BASE_URL", "http://localhost:8000"))

    def get_provider_organization(self) -> str:
        """Get the provider organization name. Default: "Health Universe" """
        return "Health Universe"

    def get_provider_url(self) -> str:
        """Get the provider URL. Default: "https://healthuniverse.com" """
        return "https://healthuniverse.com"

    def get_agent_skills(self) -> list[AgentSkill]:
        """
        Return list of skills for the AgentCard.

        Override to declare specific capabilities.

        Example:
            def get_agent_skills(self) -> list[AgentSkill]:
                return [
                    AgentSkill(
                        id="analyze_data",
                        name="Analyze Data",
                        description="Analyzes CSV datasets",
                        tags=["analysis", "csv", "data"],
                    )
                ]
        """
        return []

    def get_security_schemes(self) -> dict[str, Any]:
        """Get security schemes for the AgentCard. Default: empty"""
        return {}

    def get_security_requirements(self) -> list[dict[str, list[str]]]:
        """Get security requirements for the AgentCard. Default: empty"""
        return []

    def supports_streaming(self) -> bool:
        """Whether this agent supports streaming responses. Default: False"""
        return False

    def supports_push_notifications(self) -> bool:
        """Whether this agent supports push notifications. Default: False"""
        return False

    # Optional tools and LLM integration

    def get_tools(self) -> list[Any]:
        """
        Return list of tools for tool-based agents.

        Tools should be langchain_core.tools.Tool instances or compatible objects.
        """
        return []

    def get_system_instruction(self) -> str:
        """Return system instruction for LLM-based agents."""
        return "You are a helpful AI assistant."

    # Inter-agent communication

    async def call_agent(
        self,
        agent_name_or_url: str,
        message: str | dict | list | Any,
        context: BaseContext | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """
        Unified method to call another A2A agent.

        Args:
            agent_name_or_url: Target agent - can be:
                - Agent name (resolved via registry)
                - Full URL (http://... or https://...)
                - Local path (/processor - resolved to local base URL)
            message: Message to send (string, dict, or list)
            context: Optional context for JWT propagation
            timeout: Request timeout in seconds (default: 30s)

        Returns:
            Parsed response data

        Example:
            # Call with text message
            result = await self.call_agent("section_writer", "Write section 1")

            # Call with structured data
            result = await self.call_agent(
                "processor",
                {"document_path": "/path/to/doc.pdf"}
            )

            # Call with context for JWT propagation
            result = await self.call_agent(
                "analyzer",
                {"query": "analyze"},
                context=context,
                timeout=60.0
            )
        """
        auth_token = context.auth_token if context else None

        if agent_name_or_url.startswith(("http://", "https://", "/")):
            client = InterAgentClient(
                agent_identifier=agent_name_or_url,
                auth_token=auth_token,
                timeout=timeout,
            )
        else:
            client = InterAgentClient.from_registry(
                agent_name=agent_name_or_url,
                auth_token=auth_token,
                timeout=timeout,
            )

        try:
            if isinstance(message, str):
                response = await client.call(message, timeout=timeout)
            else:
                response = await client.call_with_data(message, timeout=timeout)

            return self._parse_agent_response(response)
        finally:
            await client.close()

    def _parse_agent_response(self, response: AgentResponse) -> Any:
        """Parse agent response into most useful form."""
        raw = response.raw_response

        # Artifact with parts
        if "artifactId" in raw and "parts" in raw:
            parts = raw["parts"]
            if parts and isinstance(parts, list):
                first_part = parts[0]
                if isinstance(first_part, dict) and "data" in first_part:
                    return first_part["data"]
                if isinstance(first_part, dict) and "text" in first_part:
                    return first_part["text"]

        # Response with artifacts array
        if "artifacts" in raw:
            artifacts = raw["artifacts"]
            if artifacts and isinstance(artifacts, list):
                first_artifact = artifacts[0]
                if isinstance(first_artifact, dict) and "parts" in first_artifact:
                    parts = first_artifact["parts"]
                    if parts:
                        first_part = parts[0]
                        if isinstance(first_part, dict) and "data" in first_part:
                            return first_part["data"]

        # Data at root
        if response.data is not None:
            return response.data

        # Text response
        if response.text:
            return response.text

        return raw

    async def call_other_agent(
        self,
        agent_identifier: str,
        message: str,
        context: BaseContext,
        timeout: float = 30.0,
    ) -> AgentResponse:
        """
        Call another A2A-compliant agent with JWT propagation.

        Args:
            agent_identifier: Target agent ("/processor", "https://...", or registry name)
            message: Message to send
            context: Message context (provides auth token)
            timeout: Request timeout in seconds

        Returns:
            AgentResponse with text, data, parts, and raw_response properties

        Example:
            response = await self.call_other_agent("/processor", message, context)
            processed = response.text
        """
        client = context.create_inter_agent_client(agent_identifier, timeout=timeout)
        try:
            return await client.call(message, timeout=timeout)
        finally:
            await client.close()

    async def call_other_agent_with_data(
        self,
        agent_identifier: str,
        data: Any,
        context: BaseContext,
        timeout: float = 30.0,
    ) -> AgentResponse:
        """
        Call another A2A-compliant agent with structured data.

        Args:
            agent_identifier: Target agent ("/processor", "https://...", or registry name)
            data: Structured data (dict, list, etc.) to send
            context: Message context (provides auth token)
            timeout: Request timeout in seconds

        Returns:
            AgentResponse with text, data, parts, and raw_response properties

        Example:
            response = await self.call_other_agent_with_data(
                "/data-processor",
                {"query": message, "format": "json"},
                context
            )
        """
        client = context.create_inter_agent_client(agent_identifier, timeout=timeout)
        try:
            return await client.call_with_data(data, timeout=timeout)
        finally:
            await client.close()

    # Server utilities

    def serve(
        self,
        host: str | None = None,
        port: int | None = None,
        reload: bool | None = None,
        log_level: str = "info",
    ) -> None:
        """
        Start an HTTP server for this agent.

        Environment variables:
            HOST: Server host (default: "0.0.0.0")
            PORT or AGENT_PORT: Server port (default: 8000)
            RELOAD: Enable auto-reload (default: "false")

        Args:
            host: Server host (overrides env var)
            port: Server port (overrides env var)
            reload: Enable auto-reload (overrides env var)
            log_level: Uvicorn log level (default: "info")

        Example:
            if __name__ == "__main__":
                agent = MyAgent()
                agent.serve()  # Starts server on http://0.0.0.0:8000
        """
        from health_universe_a2a.server import serve

        serve(self, host=host, port=port, reload=reload, log_level=log_level)

    # Optional lifecycle hooks

    async def on_startup(self) -> None:  # noqa: B027
        """
        Called when agent starts up.

        Use for: loading models, initializing connections, warming caches

        Example:
            async def on_startup(self) -> None:
                self.model = await load_model()
        """
        pass

    async def on_shutdown(self) -> None:  # noqa: B027
        """
        Called when agent shuts down.

        Use for: closing connections, saving state, cleanup

        Example:
            async def on_shutdown(self) -> None:
                await self.db.close()
        """
        pass
