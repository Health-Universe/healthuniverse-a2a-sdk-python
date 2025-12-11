"""Base A2A Agent class - simplified for library usage"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

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
from health_universe_a2a.types.extensions import (
    FILE_ACCESS_EXTENSION_URI,
    FILE_ACCESS_EXTENSION_URI_V2,
)
from health_universe_a2a.types.validation import (
    ValidationAccepted,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class A2AAgentBase(AgentExecutor, ABC):
    """
    Base class providing shared configuration and utilities for A2A agents.

    This class contains all shared functionality like agent card creation,
    validation, inter-agent communication, and lifecycle hooks.

    Execution methods (process_message, handle_request) are defined by subclasses:
    - StreamingAgent: For short-running tasks with SSE streaming
    - AsyncAgent: For long-running background tasks with POST updates

    Subclasses must implement:
    - get_agent_name()
    - get_agent_description()
    - process_message() (with their specific context type)
    - handle_request() (with their specific execution pattern)

    Optional overrides:
    - validate_message() - Validate incoming messages
    - requires_file_access() - Enable file access extension
    - get_agent_version() - Custom version string
    - get_supported_input_formats() - Supported input MIME types
    - get_supported_output_formats() - Supported output MIME types
    - Lifecycle hooks: on_startup(), on_shutdown(), etc.
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
    async def handle_request(
        self, message: str, context: StreamingContext, metadata: dict[str, Any]
    ) -> str | None:
        """
        Handle an incoming request with a StreamingContext.

        This is called by execute() and must be implemented by subclasses.

        For StreamingAgent:
            - Validate the message
            - Process it via process_message()
            - Return the result

        For AsyncAgent:
            - Validate the message via StreamingContext (SSE)
            - Send acknowledgment via StreamingContext
            - Spawn background task with BackgroundContext
            - Return acknowledgment message

        Args:
            message: The extracted message text
            context: StreamingContext for SSE updates
            metadata: Request metadata

        Returns:
            Result string (for StreamingAgent) or ack message (for AsyncAgent),
            or None if validation rejected
        """
        pass

    # AgentExecutor interface implementation (shared by all agent types)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute agent processing for a request.

        This implements the AgentExecutor interface required by a2a-sdk's DefaultRequestHandler.
        It:
        1. Creates a TaskUpdater from the event_queue
        2. Extracts message text and metadata from RequestContext
        3. Builds a StreamingContext for the agent
        4. Calls agent.handle_request()
        5. Sends completion message if handle_request returns a result

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
                # User object has user_name attribute
                user_id = getattr(context.call_context.user, "user_name", None)

            # Extract metadata from context (metadata includes extension parameters)
            metadata = context.metadata or {}

            # Extract thread_id from metadata (A2A spec doesn't mandate thread_id in call_context)
            if "thread_id" in metadata:
                thread_id = metadata["thread_id"]

            # Extract file access token from extensions (v2 preferred, v1 fallback)
            file_access_token = None
            if FILE_ACCESS_EXTENSION_URI_V2 in metadata:
                file_ext_data = metadata[FILE_ACCESS_EXTENSION_URI_V2]
                if isinstance(file_ext_data, dict):
                    file_access_token = file_ext_data.get("access_token")
            elif FILE_ACCESS_EXTENSION_URI in metadata:
                # Legacy v1 fallback
                file_ext_data = metadata[FILE_ACCESS_EXTENSION_URI]
                if isinstance(file_ext_data, dict):
                    file_access_token = file_ext_data.get("access_token")

            # Extract extensions list from message
            extensions = None
            if context.message and hasattr(context.message, "extensions"):
                extensions = context.message.extensions

            # Extract JWT auth token for inter-agent calls
            # Check common locations where the token might be stored
            auth_token = None
            if context.call_context and context.call_context.state:
                # Check for authorization in call_context.state (common location)
                auth_token = context.call_context.state.get("authorization")
                # Also try lowercase and with Bearer prefix
                if not auth_token:
                    auth_header = context.call_context.state.get("Authorization")
                    if auth_header and isinstance(auth_header, str):
                        # Remove "Bearer " prefix if present
                        auth_token = auth_header.replace("Bearer ", "").strip()

            # Build StreamingContext for agent
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

            # Call agent's handle_request (this is the main agent logic)
            logger.info("Calling agent.handle_request()")
            result = await self.handle_request(
                message=message_text, context=message_context, metadata=metadata
            )

            logger.info(f"Agent returned result: {result is not None}")

            # For StreamingAgent: Send completion with result
            # For AsyncAgent: Already sent "submitted" status with final=True,
            #                 completion happens in background via POST
            # Check if this is background mode by checking push_notifications support
            is_background_mode = self.supports_push_notifications()

            if result and not is_background_mode:
                # StreamingAgent: send final message with result
                logger.info("Sending final completion message for streaming agent")
                text_part = TextPart(text=result)
                final_message = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(root=text_part)],
                )
                await updater.complete(message=final_message)
            elif result and is_background_mode:
                # AsyncAgent: ack already sent with final=True, background task running
                logger.info(
                    "AsyncAgent: ack sent via SSE, background processing running with POST updates"
                )

        except Exception as error:
            logger.error(f"Agent execution failed: {error}", exc_info=True)

            # Send failure message
            error_text = f"Agent execution failed: {str(error)}"
            error_part = TextPart(text=error_text)
            error_message = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=error_part)],
            )
            await updater.failed(message=error_message)

            # Re-raise so server can handle it
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel the agent execution.

        **IMPORTANT: Cancellation is not fully implemented in this SDK.**

        This method is called when a task cancellation is requested by the A2A
        protocol, but the default implementation does not:
        - Set any cancellation flags on the context
        - Stop or interrupt running tasks
        - Provide any automatic cancellation behavior

        Current limitations:
        - context.is_cancelled() will always return False
        - The _cancelled flag is never set to True
        - Agents must implement their own cancellation logic if needed

        For production use, consider:
        1. Override this method in your agent subclass
        2. Store cancellation state (e.g., in a shared dict keyed by task_id)
        3. Check that state periodically in your process_message() implementation
        4. For AsyncAgent background tasks, use asyncio.CancelledError handling

        Args:
            context: Request context for the task to cancel
            event_queue: Event queue for sending cancellation updates

        Example implementation:
            # In your agent class:
            def __init__(self):
                super().__init__()
                self._cancelled_tasks = set()

            async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
                self._cancelled_tasks.add(context.task_id)
                logger.info(f"Marked task {context.task_id} as cancelled")

            async def process_message(self, message: str, context: StreamingContext) -> str:
                if context.task_id in self._cancelled_tasks:
                    return "Task was cancelled"
                # ... rest of processing
        """
        logger.info(f"Cancelling execution for {self.get_agent_name()} task_id={context.task_id}")
        # Default implementation does nothing - cancellation not fully implemented

    def _extract_message_text(self, message: Message) -> str:
        """
        Extract text content from an A2A Message.

        Args:
            message: The A2A Message object

        Returns:
            Concatenated text from all text parts
        """
        text_parts: list[str] = []

        for part in message.parts:
            if part.root and hasattr(part.root, "text"):
                text_parts.append(part.root.text)
            elif hasattr(part, "text"):
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else ""

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
                extensions=self.get_extensions(),
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

    async def call_agent(
        self,
        agent_name_or_url: str,
        message: str | dict | list | Any,
        context: BaseContext | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """
        Unified method to call another A2A agent.

        This method handles all message types (string, dict, list) and
        automatically parses the response. It matches the pattern used
        in the example agent repos.

        Args:
            agent_name_or_url: Target agent - can be:
                - Agent name (resolved via registry)
                - Full URL (http://... or https://...)
                - Local path (/processor - resolved to local base URL)
            message: Message to send - can be:
                - String: sent as TextPart
                - Dict/List: sent as DataPart
                - Pre-formatted with 'parts': sent as-is
            context: Optional context for JWT propagation
            timeout: Request timeout in seconds (default: 30s)

        Returns:
            Parsed response data. Returns the most useful form:
            - For artifact responses: the artifact data/content
            - For message responses: the message text or data part
            - Falls back to raw response if parsing fails

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
        # Create client with registry support
        auth_token = context.auth_token if context else None

        if agent_name_or_url.startswith(("http://", "https://", "/")):
            # Direct URL or local path
            client = InterAgentClient(
                agent_identifier=agent_name_or_url,
                auth_token=auth_token,
                timeout=timeout,
            )
        else:
            # Registry lookup
            client = InterAgentClient.from_registry(
                agent_name=agent_name_or_url,
                auth_token=auth_token,
                timeout=timeout,
            )

        try:
            # Determine message type and call appropriately
            if isinstance(message, str):
                response = await client.call(message, timeout=timeout)
            else:
                response = await client.call_with_data(message, timeout=timeout)

            # Parse and return the most useful form
            return self._parse_agent_response(response)
        finally:
            await client.close()

    def _parse_agent_response(self, response: AgentResponse) -> Any:
        """
        Parse agent response into most useful form.

        Handles multiple response formats:
        1. Artifact with artifactId and parts
        2. Full response with artifacts array
        3. Direct data at root level
        4. Message with text parts

        Args:
            response: AgentResponse from inter-agent call

        Returns:
            Parsed data in most useful form
        """
        raw = response.raw_response

        # Case 1: Artifact with artifactId and parts (common pattern)
        if isinstance(raw, dict):
            if "artifactId" in raw and "parts" in raw:
                parts = raw["parts"]
                if parts and isinstance(parts, list):
                    first_part = parts[0]
                    if isinstance(first_part, dict) and "data" in first_part:
                        return first_part["data"]
                    if isinstance(first_part, dict) and "text" in first_part:
                        return first_part["text"]

            # Case 2: Full response with artifacts array
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

            # Case 3: Data at root (direct structured response)
            if response.data is not None:
                return response.data

        # Case 4: Text response
        if response.text:
            return response.text

        # Fallback: return raw response
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
            async def process_message(self, message: str, context: StreamingContext) -> str:
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
        context: BaseContext,
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
            async def process_message(self, message: str, context: StreamingContext) -> str:
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

                async def process_message(self, message: str, context: StreamingContext) -> str:
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
