"""Base A2A Agent class - simplified for library usage"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
    TextPart,
)

from health_universe_a2a import AgentResponse
from health_universe_a2a.context import AsyncContext, MessageContext
from health_universe_a2a.types.extensions import AgentExtension
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# TypeVar for context, bound to MessageContext (which AsyncContext extends)
ContextT = TypeVar("ContextT", bound=MessageContext)


class A2AAgent(ABC, Generic[ContextT]):
    """
    Base class for all A2A agents.

    This is a simplified interface for building A2A-compliant agents.
    Subclasses must implement:
    - get_agent_name()
    - get_agent_description()
    - process_message()

    Optional overrides:
    - validate_message() - Validate incoming messages
    - requires_file_access() - Enable file access extension
    - get_agent_version() - Custom version string
    - get_supported_input_formats() - Supported input MIME types
    - get_supported_output_formats() - Supported output MIME types
    - Lifecycle hooks: on_startup(), on_shutdown(), etc.

    Example:
        class MyAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "My Agent"

            def get_agent_description(self) -> str:
                return "Does something useful"

            async def process_message(self, message: str, context: MessageContext) -> str:
                await context.update_progress("Working...", 0.5)
                result = await self.do_work(message)
                return f"Done! Result: {result}"
    """

    def __init__(self) -> None:
        """Initialize the agent with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)

    # Required abstract methods

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the agent's name for the AgentCard.

        Returns:
            Agent name (e.g., "Data Analyzer")
        """
        pass

    @abstractmethod
    def get_agent_description(self) -> str:
        """
        Return the agent's description for the AgentCard.

        Returns:
            Agent description (e.g., "Analyzes datasets and returns insights")
        """
        pass

    @abstractmethod
    async def process_message(self, message: str, context: ContextT) -> str:
        """
        Process an incoming message and return a response.

        Use context methods to send progress updates and artifacts during processing.

        Args:
            message: The extracted text message to process
            context: Context with update methods and metadata

        Returns:
            Final response message as string

        Example:
            async def process_message(self, message: str, context: MessageContext) -> str:
                await context.update_progress("Loading...", 0.2)
                data = await self.load_data()

                await context.update_progress("Processing...", 0.6)
                result = await self.process(data)

                await context.add_artifact("Results", json.dumps(result))
                return "Processing complete!"
        """
        pass

    # Request handling with validation

    async def handle_request(
        self, message: str, context: ContextT, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle complete request flow: validation → processing.

        This method orchestrates the full A2A request flow:
        1. Validates the message via validate_message()
        2. If rejected: Sends error update and returns None
        3. If accepted: Calls lifecycle hooks and process_message()

        The SDK handles sending appropriate A2A task updates based on validation result.

        Args:
            message: The message to process
            context: Message context for sending updates
            metadata: Request metadata for validation

        Returns:
            Final response string if accepted and processing succeeds, None if rejected

        Note: This is called by the A2A server integration (NestJS backend).
        You typically don't need to override this - override validate_message() and
        process_message() instead.
        """
        # Step 1: Validate
        validation_result = await self.validate_message(message, metadata)

        # Step 2: Handle rejection
        if isinstance(validation_result, ValidationRejected):
            self.logger.warning(f"Message validation failed: {validation_result.reason}")
            # Send rejection via A2A protocol using TaskUpdater.reject()
            if context._updater:
                # Create rejection message
                text_part = TextPart(text=f"Validation failed: {validation_result.reason}")
                msg = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(root=text_part)],
                )

                # Reject the task
                await context._updater.reject(message=msg)
            return None

        # Step 3: Handle acceptance
        if isinstance(validation_result, ValidationAccepted):
            self.logger.info(
                "Message validation passed"
                + (
                    f" (estimated: {validation_result.estimated_duration_seconds}s)"
                    if validation_result.estimated_duration_seconds
                    else ""
                )
            )

            # Call lifecycle hook
            await self.on_task_start(message, context)

            try:
                # Process the message
                result = await self.process_message(message, context)

                # Call completion hook
                await self.on_task_complete(message, result, context)

                return result

            except Exception as error:
                # Call error hook
                custom_error = await self.on_task_error(message, error, context)
                error_msg = custom_error or f"Processing error: {str(error)}"

                # Send error update
                if context._updater:
                    await context._updater.update_progress(
                        message=error_msg, progress=None, status="error"
                    )

                # Re-raise for server to handle
                raise

        return None

    # Optional validation hook

    async def validate_message(self, message: str, metadata: dict[str, Any]) -> ValidationResult:
        """
        Validate incoming message before processing.

        This runs BEFORE the task is enqueued (for background agents)
        or BEFORE processing starts (for realtime agents).

        Use this to:
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
        Override to add custom validation logic.

        Example:
            from health_universe_a2a import ValidationAccepted, ValidationRejected

            async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
                if len(message) < 10:
                    return ValidationRejected(
                        reason="Message too short (min 10 chars)"
                    )
                return ValidationAccepted(estimated_duration_seconds=60)
        """
        return ValidationAccepted()

    # Optional configuration methods

    def get_agent_version(self) -> str:
        """
        Return agent version.

        Returns:
            Version string (default: "1.0.0")
        """
        return "1.0.0"

    def requires_file_access(self) -> bool:
        """
        Whether agent needs file access extension.

        When True, the file access extension will be automatically
        added to the agent card, and context.file_access_token
        will be populated.

        Returns:
            True if file access is required, False otherwise (default: False)
        """
        return False

    def get_supported_input_formats(self) -> list[str]:
        """
        Supported input MIME types.

        Returns:
            List of MIME types (default: ["text/plain", "application/json"])
        """
        return ["text/plain", "application/json"]

    def get_supported_output_formats(self) -> list[str]:
        """
        Supported output MIME types.

        Returns:
            List of MIME types (default: ["text/plain", "application/json"])
        """
        return ["text/plain", "application/json"]

    def get_extensions(self) -> list[AgentExtension]:
        """
        Return list of extensions supported by this agent.

        This is automatically configured by StreamingAgent and AsyncAgent.
        Only override if you need custom extensions.

        Returns:
            List of AgentExtension objects
        """
        return []

    # AgentCard creation and configuration

    def create_agent_card(self) -> AgentCard:
        """
        Create the AgentCard for agent discovery.
        Fully compliant with A2A specification v0.3.0.

        This method automatically builds the AgentCard using all the overridable
        configuration methods (get_agent_name(), get_agent_description(), etc.).

        Returns:
            AgentCard with agent metadata and capabilities

        Example:
            # The agent card is automatically created from your configuration:
            agent = MyAgent()
            card = agent.create_agent_card()
            # Card includes name, description, version, capabilities, skills, etc.
        """
        # Get base URL from environment variable
        base_url = self.get_base_url()

        return AgentCard(
            # Required fields per spec
            protocol_version="0.3.0",
            name=self.get_agent_name(),
            description=self.get_agent_description(),
            version=self.get_agent_version(),
            url=base_url,
            # Transport configuration - using JSON-RPC
            preferred_transport="JSONRPC",
            additional_interfaces=[AgentInterface(url=base_url, transport="JSONRPC")],
            # Provider information
            provider=AgentProvider(
                organization=self.get_provider_organization(), url=self.get_provider_url()
            ),
            # Capabilities
            capabilities=AgentCapabilities(
                streaming=self.supports_streaming(),
                push_notifications=self.supports_push_notifications(),
                extensions=self.get_extensions(),  # type: ignore[arg-type]
            ),
            # Skills (optional but recommended)
            skills=self.get_agent_skills(),
            # Security (can be extended in subclasses)
            security_schemes=self.get_security_schemes(),
            security=self.get_security_requirements(),
            # Input/output modes
            default_input_modes=self.get_supported_input_formats(),
            default_output_modes=self.get_supported_output_formats(),
        )

    def get_base_url(self) -> str:
        """
        Get the agent's base URL.

        Returns the URL from HU_APP_URL or A2A_BASE_URL environment variable.
        Override to provide a custom base URL.

        Returns:
            Base URL string (default: from environment or "http://localhost:8000")
        """
        return os.getenv("HU_APP_URL", os.getenv("A2A_BASE_URL", "http://localhost:8000"))

    def get_provider_organization(self) -> str:
        """
        Get the provider organization name.

        Override to specify your organization name.

        Returns:
            Organization name (default: "Health Universe")
        """
        return "Health Universe"

    def get_provider_url(self) -> str:
        """
        Get the provider URL.

        Override to specify your organization's URL.

        Returns:
            Provider URL (default: "https://healthuniverse.com")
        """
        return "https://healthuniverse.com"

    def get_agent_skills(self) -> list[AgentSkill]:
        """
        Return list of skills for the AgentCard.

        Skills define specific capabilities of the agent in granular detail.
        Each skill can specify its own inputModes/outputModes if they
        differ from the agent's defaults.

        Override to declare specific capabilities.

        Returns:
            List of AgentSkill objects or empty list (default: empty)

        Example:
            def get_agent_skills(self) -> list[AgentSkill]:
                return [
                    AgentSkill(
                        id="analyze_data",
                        name="Analyze Data",
                        description="Analyzes CSV datasets",
                        tags=["analysis", "csv", "data"],
                        input_modes=["text/csv"],
                        output_modes=["application/json"],
                        examples=["file://data.csv"]
                    ),
                    AgentSkill(
                        id="generate_report",
                        name="Generate Report",
                        description="Generates summary reports",
                        tags=["reporting", "documentation"],
                        input_modes=["application/json"],
                        output_modes=["text/html", "application/pdf"],
                        examples=['{"data": [1, 2, 3]}']
                    )
                ]
        """
        return []

    def get_security_schemes(self) -> dict[str, Any]:
        """
        Get security schemes for the AgentCard.

        Override to specify authentication/authorization schemes.

        Returns:
            Security schemes dict (default: empty)

        Example:
            def get_security_schemes(self) -> dict[str, Any]:
                return {
                    "oauth": {
                        "type": "openIdConnect",
                        "openIdConnectUrl": "https://example.com/.well-known/openid-configuration"
                    }
                }
        """
        return {}

    def get_security_requirements(self) -> list[dict[str, list[str]]]:
        """
        Get security requirements for the AgentCard.

        Override to specify required security schemes and scopes.

        Returns:
            List of security requirements (default: empty)

        Example:
            def get_security_requirements(self) -> list[dict[str, list[str]]]:
                return [{"oauth": ["openid", "profile", "email"]}]
        """
        return []

    def supports_streaming(self) -> bool:
        """
        Whether this agent supports streaming responses.

        Override to enable streaming support. This is automatically
        set to True by StreamingAgent and False by AsyncAgent.

        Returns:
            True if streaming is supported, False otherwise (default: False)
        """
        return False

    def supports_push_notifications(self) -> bool:
        """
        Whether this agent supports push notifications.

        Override to enable push notification support. This is automatically
        set to True by AsyncAgent and False by StreamingAgent.

        Returns:
            True if push notifications are supported, False otherwise (default: False)
        """
        return False

    # Optional tools and LLM integration

    def get_tools(self) -> list[Any]:
        """
        Return list of tools for tool-based agents.

        Tools should be langchain_core.tools.Tool instances or compatible objects.
        The integration layer will handle tool execution with an LLM.

        Override to provide tool-based functionality.

        Returns:
            List of tools or empty list for non-tool agents (default: empty)

        Example:
            from langchain_core.tools import Tool

            def get_tools(self) -> list[Any]:
                return [
                    Tool(
                        name="search",
                        description="Search for information",
                        func=self.search
                    ),
                    Tool(
                        name="calculate",
                        description="Perform calculations",
                        func=self.calculate
                    )
                ]
        """
        return []

    def get_system_instruction(self) -> str:
        """
        Return system instruction for LLM-based agents.

        This instruction guides the LLM's behavior when using tools
        or generating responses.

        Override to provide custom instructions for LLM agents.

        Returns:
            System instruction string (default: generic assistant instruction)

        Example:
            def get_system_instruction(self) -> str:
                return '''You are a medical data analyst assistant.
                You help users analyze healthcare datasets and generate insights.
                Always cite sources and explain your reasoning.
                If uncertain, ask for clarification.'''
        """
        return "You are a helpful AI assistant."

    # Inter-agent communication

    async def call_other_agent(
        self,
        agent_identifier: str,
        message: str,
        context: ContextT,
        timeout: float = 30.0,
    ) -> AgentResponse:
        """
        Call another A2A-compliant agent with JWT propagation.

        This is a convenience method that creates an InterAgentClient and calls it.
        The JWT from the original request is automatically propagated.

        Args:
            agent_identifier: Target agent:
                - "/processor" for local agents (same pod, no ingress)
                - "https://api.example.com/agent" for remote agents
                - "agent-name" for registry lookup (if configured)
            message: Message to send to the agent
            context: Message context (provides auth token)
            timeout: Request timeout in seconds (default: 30s, increase for slow agents)

        Returns:
            AgentResponse with text, data, parts, and raw_response properties

        Raises:
            httpx.HTTPError: If the request fails after retries
            ValueError: If agent identifier cannot be resolved

        Example:
            async def process_message(self, message: str, context: MessageContext) -> str:
                # Call local agent (bypasses ingress/egress)
                response = await self.call_other_agent("/processor", message, context)
                processed = response.text

                # Call remote agent with longer timeout
                response = await self.call_other_agent(
                    "https://api.example.com/agent",
                    processed,
                    context,
                    timeout=120.0
                )

                return response.text
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
        context: ContextT,
        timeout: float = 30.0,
    ) -> "AgentResponse":
        """
        Call another A2A-compliant agent with structured data and JWT propagation.

        This is a convenience method that properly formats structured data using DataPart.
        The JWT from the original request is automatically propagated.

        Args:
            agent_identifier: Target agent:
                - "/processor" for local agents (same pod)
                - "https://api.example.com/agent" for remote agents
                - "agent-name" for registry lookup (if configured)
            data: Structured data (dict, list, etc.) to send
            context: Message context (provides auth token)
            timeout: Request timeout in seconds (default: 30s, increase for slow agents)

        Returns:
            AgentResponse with text, data, parts, and raw_response properties

        Raises:
            httpx.HTTPError: If the request fails after retries
            ValueError: If agent identifier cannot be resolved

        Example:
            async def process_message(self, message: str, context: MessageContext) -> str:
                # Call with structured data
                response = await self.call_other_agent_with_data(
                    "/data-processor",
                    {"query": message, "format": "json"},
                    context,
                    timeout=60.0
                )

                # Access structured response
                if response.data:
                    return json.dumps(response.data)
                else:
                    return response.text
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

        This is a convenience method that starts a production-ready A2A-compliant
        HTTP server with all required endpoints.

        Environment variables:
            HOST: Server host (default: "0.0.0.0")
            PORT or AGENT_PORT: Server port (default: 8000)
            RELOAD: Enable auto-reload on code changes (default: "false")

        Args:
            host: Server host (overrides env var)
            port: Server port (overrides env var)
            reload: Enable auto-reload (overrides env var)
            log_level: Uvicorn log level (default: "info")

        Raises:
            ImportError: If uvicorn or a2a server dependencies are not installed

        Example:
            class MyAgent(A2AAgent):
                def get_agent_name(self) -> str:
                    return "My Agent"

                def get_agent_description(self) -> str:
                    return "Does something useful"

                async def process_message(self, message: str, context: MessageContext) -> str:
                    return f"Processed: {message}"

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
                self.db = await connect_to_db()
        """
        pass

    async def on_shutdown(self) -> None:  # noqa: B027
        """
        Called when agent shuts down.

        Use for: closing connections, saving state, cleanup

        Example:
            async def on_shutdown(self) -> None:
                await self.db.close()
                await self.model.unload()
        """
        pass

    async def on_task_start(self, message: str, context: ContextT) -> None:  # noqa: B027
        """
        Called after validation passes, before process_message.

        Use for: logging, metrics, setup

        Args:
            message: The message being processed
            context: Message context

        Example:
            async def on_task_start(self, message: str, context: MessageContext) -> None:
                self.logger.info(f"Starting task for user {context.user_id}")
                await self.metrics.increment("tasks_started")
        """
        pass

    async def on_task_complete(
        self, message: str, result: str, context: ContextT
    ) -> None:  # noqa: B027
        """
        Called after process_message completes successfully.

        Use for: logging, metrics, cleanup

        Args:
            message: The message that was processed
            result: The result returned by process_message
            context: Message context

        Example:
            async def on_task_complete(
                self, message: str, result: str, context: MessageContext
            ) -> None:
                self.logger.info(f"Task completed for user {context.user_id}")
                await self.metrics.increment("tasks_completed")
        """
        pass

    async def on_task_error(
        self, message: str, error: Exception, context: ContextT
    ) -> str | None:
        """
        Called when process_message raises an exception.

        Use for: error logging, cleanup, custom error handling

        Args:
            message: The message being processed
            error: The exception that was raised
            context: Message context

        Returns:
            Optional custom error message to override default
            (return None to use default error message)

        Example:
            async def on_task_error(
                self, message: str, error: Exception, context: MessageContext
            ) -> str | None:
                self.logger.error(f"Task failed: {error}")
                if isinstance(error, TimeoutError):
                    return "Processing timed out. Please try a smaller request."
                return None  # Use default error message
        """
        return None
