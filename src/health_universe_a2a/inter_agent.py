"""Inter-agent communication client with JWT propagation and retry logic"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from a2a.types import DataPart, Message, Part, Role, TextPart

if TYPE_CHECKING:
    from health_universe_a2a.inspect_ai.logger import InspectLogger

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for resolving agent names to URLs.

    Supports loading from:
    - Environment variable (AGENT_REGISTRY as JSON string)
    - Configuration file (AGENT_REGISTRY_PATH as path to JSON file)

    Example config file (agents.json):
        {
            "agents": {
                "protocol_orchestrator": {
                    "url": "http://localhost:8010",
                    "name": "Protocol Orchestrator"
                },
                "section_writer": {
                    "url": "http://localhost:8002",
                    "name": "Section Writer"
                }
            }
        }

    Example usage:
        registry = AgentRegistry()
        url = registry.resolve_url("section_writer")
        # Returns: "http://localhost:8002"
    """

    def __init__(self, registry_path: str | None = None):
        """
        Initialize agent registry.

        Args:
            registry_path: Path to agents.json file. If not provided, checks
                AGENT_REGISTRY_PATH environment variable.
        """
        self.registry_path = registry_path or os.getenv("AGENT_REGISTRY_PATH")
        self._cache: dict[str, Any] | None = None
        self._flat_cache: dict[str, str] | None = None

    def load(self) -> dict[str, Any]:
        """
        Load and cache registry configuration.

        Tries in order:
        1. Registry file at registry_path or AGENT_REGISTRY_PATH
        2. AGENT_REGISTRY environment variable (JSON string)
        3. Empty dict if neither available

        Returns:
            Registry configuration dict
        """
        if self._cache is not None:
            return self._cache

        # Try loading from file
        if self.registry_path:
            path = Path(self.registry_path)
            if path.exists():
                try:
                    with open(path) as f:
                        loaded = cast(dict[str, Any], json.load(f))
                        self._cache = loaded
                        logger.debug(f"Loaded agent registry from {path}")
                        return loaded
                except Exception as e:
                    logger.warning(f"Failed to load agent registry from {path}: {e}")

        # Try loading from environment variable
        registry_json = os.getenv("AGENT_REGISTRY")
        if registry_json:
            try:
                loaded = cast(dict[str, Any], json.loads(registry_json))
                self._cache = loaded
                logger.debug("Loaded agent registry from AGENT_REGISTRY env var")
                return loaded
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AGENT_REGISTRY env var: {e}")

        # Return empty dict
        self._cache = {}
        return self._cache

    def _build_flat_registry(self) -> dict[str, str]:
        """Build flat agent_name -> URL mapping from config."""
        if self._flat_cache is not None:
            return self._flat_cache

        config = self.load()
        self._flat_cache = {}

        # Handle nested agents structure
        if "agents" in config:
            for name, info in config["agents"].items():
                if isinstance(info, dict) and "url" in info:
                    self._flat_cache[name] = info["url"]
                elif isinstance(info, str):
                    self._flat_cache[name] = info
        # Handle flat structure (config is already dict[str, Any])
        else:
            for name, value in config.items():
                if isinstance(value, str):
                    self._flat_cache[name] = value
                elif isinstance(value, dict) and "url" in value:
                    self._flat_cache[name] = value["url"]

        return self._flat_cache

    def resolve_url(self, agent_name: str) -> str:
        """
        Resolve agent name to URL.

        Args:
            agent_name: Name of the agent to resolve

        Returns:
            URL for the agent

        Raises:
            ValueError: If agent name not found in registry
        """
        registry = self._build_flat_registry()

        if agent_name in registry:
            return registry[agent_name]

        raise ValueError(
            f"Agent '{agent_name}' not found in registry. Available agents: {list(registry.keys())}"
        )

    def clear_cache(self) -> None:
        """Clear cached registry data (useful for testing or reloading)."""
        self._cache = None
        self._flat_cache = None

    def get_agent_info(self, agent_name: str) -> dict[str, Any]:
        """
        Get full agent info (url, name, description, etc.).

        Args:
            agent_name: Name of the agent

        Returns:
            Agent info dict with at least 'url' key

        Raises:
            ValueError: If agent not found
        """
        config = self.load()

        if "agents" in config and agent_name in config["agents"]:
            info = config["agents"][agent_name]
            if isinstance(info, str):
                return {"url": info}
            return dict(info) if isinstance(info, dict) else {"url": str(info)}

        if agent_name in config:
            info = config[agent_name]
            if isinstance(info, str):
                return {"url": info}
            return dict(info) if isinstance(info, dict) else {"url": str(info)}

        raise ValueError(f"Agent '{agent_name}' not found in registry")


# Global registry instance
_global_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get or create global agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


class AgentResponse:
    """
    Response from an inter-agent call.

    Provides both raw A2A response and convenient parsed data access.
    """

    def __init__(self, raw_response: dict[str, Any]):
        """
        Initialize from raw A2A response.

        Args:
            raw_response: Raw JSON response from agent
        """
        self.raw_response = raw_response
        self._text_cache: str | None = None
        self._data_cache: Any | None = None

    @property
    def text(self) -> str:
        """
        Extract text from response parts.

        Concatenates all text parts with newlines.
        For simple text responses, this is the most convenient access method.

        Returns:
            Extracted text or empty string if no text parts
        """
        if self._text_cache is not None:
            return self._text_cache

        texts = []
        if "message" in self.raw_response:
            msg = self.raw_response["message"]
            if isinstance(msg, dict) and "parts" in msg:
                for part in msg["parts"]:
                    if isinstance(part, dict):
                        kind = part.get("kind")
                        if kind == "text":
                            texts.append(part.get("text", ""))

        self._text_cache = "\n".join(texts)
        return self._text_cache

    @property
    def data(self) -> Any:
        """
        Extract first data part from response.

        For responses with structured data, this extracts the first DataPart.

        Returns:
            Extracted data or None if no data parts
        """
        if self._data_cache is not None:
            return self._data_cache

        if "message" in self.raw_response:
            msg = self.raw_response["message"]
            if isinstance(msg, dict) and "parts" in msg:
                for part in msg["parts"]:
                    if isinstance(part, dict):
                        kind = part.get("kind")
                        if kind == "data":
                            self._data_cache = part.get("data")
                            return self._data_cache

        return None

    @property
    def parts(self) -> list[dict[str, Any]]:
        """
        Get all parts from response.

        For complex responses with multiple parts, access raw parts array.
        Each part has a 'kind' field ('text', 'data', 'file').

        Returns:
            List of part dictionaries (with 'kind' field)
        """
        if "message" in self.raw_response:
            msg = self.raw_response["message"]
            if isinstance(msg, dict) and "parts" in msg:
                parts: list[dict[str, Any]] = msg["parts"]
                return parts
        return []

    def __str__(self) -> str:
        """String representation returns text."""
        return self.text

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"AgentResponse(text={self.text!r}, data={self.data!r})"


class InterAgentClient:
    """
    Client for calling other A2A-compliant agents with automatic JWT propagation.

    Supports three agent identifier formats:
    - Local agents: "/agent-path" → http://localhost:8501/agent-path
    - Direct URLs: "http://example.com/agent" → Direct call
    - Agent names: "data-processor" → Registry lookup (if configured)

    Features:
    - Automatic JWT propagation from original request
    - Local agent detection (bypasses ingress/egress)
    - Auto-retry with exponential backoff (transient errors only)
    - Structured response with raw data access
    - Configurable timeouts

    Example:
        # Via context (recommended - auto JWT propagation)
        client = context.create_inter_agent_client("/processor")
        response = await client.call("Process this data")
        print(response.text)

        # Direct instantiation
        client = InterAgentClient(
            "/processor",
            auth_token="jwt-token",
            timeout=60.0
        )
        response = await client.call_with_data({"query": "test"})
        print(response.data)
    """

    def __init__(
        self,
        agent_identifier: str,
        auth_token: str | None = None,
        local_base_url: str | None = None,
        agent_registry: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        inspect_logger: "InspectLogger | None" = None,
    ):
        """
        Initialize inter-agent client.

        Args:
            agent_identifier: Target agent (/, http(s)://, or name)
            auth_token: JWT token to propagate (optional)
            local_base_url: Base URL for local agents (default: from env or localhost:8501)
            agent_registry: Map of agent names to URLs (default: from env)
            timeout: Default timeout in seconds (default: 30.0, increase for slow agents)
            max_retries: Max retry attempts for transient errors (default: 3)
            retry_backoff_base: Base for exponential backoff in seconds (default: 1.0)
            inspect_logger: Optional InspectLogger for observability
        """
        self.agent_identifier = agent_identifier
        self.auth_token = auth_token
        base_url = local_base_url or os.getenv("LOCAL_AGENT_BASE_URL", "http://localhost:8501")
        self.local_base_url = (base_url or "http://localhost:8501").rstrip("/")
        self.agent_registry = agent_registry or self._load_agent_registry()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.target_url = self._resolve_target_url()
        self._inspect_logger = inspect_logger

    @staticmethod
    def _load_agent_registry() -> dict[str, str]:
        """
        Load agent registry from environment variable.

        Checks AGENT_REGISTRY environment variable for JSON mapping:
        AGENT_REGISTRY='{"data-processor": "http://...", "analyzer": "http://..."}'

        Returns:
            Agent name to URL mapping
        """
        registry_json = os.getenv("AGENT_REGISTRY")
        if not registry_json:
            return {}

        try:
            registry = json.loads(registry_json)
            if not isinstance(registry, dict):
                logger.warning(f"AGENT_REGISTRY is not a dict: {registry_json}")
                return {}
            return registry
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AGENT_REGISTRY: {e}")
            return {}

    @classmethod
    def from_registry(
        cls,
        agent_name: str,
        auth_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> "InterAgentClient":
        """
        Create client from registry lookup.

        Convenient factory method that resolves agent name using the global registry.

        Args:
            agent_name: Name of agent to call (must be in registry)
            auth_token: Optional JWT token to propagate
            timeout: Request timeout in seconds
            max_retries: Max retry attempts for transient errors

        Returns:
            Configured InterAgentClient

        Raises:
            ValueError: If agent name not found in registry

        Example:
            client = InterAgentClient.from_registry("section_writer")
            response = await client.call("Generate section 1")
        """
        registry = get_agent_registry()
        url = registry.resolve_url(agent_name)
        return cls(
            agent_identifier=url,
            auth_token=auth_token,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _resolve_target_url(self) -> str:
        """
        Resolve agent identifier to target URL.

        Returns:
            Full target URL

        Raises:
            ValueError: If agent cannot be resolved
        """
        identifier = self.agent_identifier

        # Local agent: /agent-path
        if identifier.startswith("/"):
            return f"{self.local_base_url}{identifier}"

        # Direct URL: http(s)://...
        if identifier.startswith(("http://", "https://")):
            return identifier

        # Agent name: lookup in registry (using global registry)
        try:
            registry = get_agent_registry()
            return registry.resolve_url(identifier)
        except ValueError:
            pass

        # Also check legacy flat registry
        if identifier in self.agent_registry:
            return self.agent_registry[identifier]

        # Not found
        raise ValueError(
            f"Could not resolve agent identifier '{identifier}'. "
            f"Use '/path' for local agents, 'http(s)://url' for direct URLs, "
            f"or register agent name in AGENT_REGISTRY environment variable."
        )

    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if error is retryable.

        Retry only transient errors (5xx, timeouts, connection errors).
        Don't retry auth/validation errors (4xx).

        Args:
            error: Exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        if isinstance(error, httpx.TimeoutException):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            # Retry 5xx server errors
            # Don't retry 4xx client errors (auth, validation, etc.)
            return error.response.status_code >= 500

        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return True

        return False

    async def _call_with_retry(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """
        Make HTTP call with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional arguments for httpx request

        Returns:
            Response from the agent

        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                async with httpx.AsyncClient(timeout=kwargs.pop("timeout", self.timeout)) as client:
                    response = await client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response

            except Exception as e:
                last_error = e

                # Don't retry if not retryable error
                if not self._should_retry(e):
                    logger.warning(f"Non-retryable error calling {url}: {e}")
                    raise

                # Don't retry on last attempt
                if attempt >= self.max_retries:
                    logger.error(f"All {self.max_retries} retries failed for {url}: {e}")
                    raise

                # Calculate backoff delay (exponential: 1s, 2s, 4s, ...)
                delay = self.retry_backoff_base * (2**attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed for {url}, "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_error or Exception("Unknown error during retry")

    async def call(self, message: str, timeout: float | None = None) -> AgentResponse:
        """
        Call the agent with a text message.

        Args:
            message: Text message to send
            timeout: Optional timeout override (default: use client timeout)

        Returns:
            AgentResponse with text, data, parts, and raw_response

        Raises:
            httpx.HTTPError: If the request fails after all retries
            ValueError: If agent identifier cannot be resolved

        Example:
            response = await client.call("Process this text")
            print(response.text)  # Convenient text access
            print(response.raw_response)  # Full A2A response
        """
        start_time = time.time()
        error_msg = None
        agent_response: AgentResponse | None = None

        try:
            # Build A2A message
            text_part = TextPart(text=message)
            a2a_message = Message(
                message_id=str(uuid.uuid4()), role=Role.user, parts=[Part(root=text_part)]
            )

            # Build request
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            request_timeout = timeout if timeout is not None else self.timeout

            # Call with retry
            response = await self._call_with_retry(
                "POST",
                f"{self.target_url}/a2a/v1/messages",
                json=a2a_message.model_dump(by_alias=True),
                headers=headers,
                timeout=request_timeout,
            )

            # Parse and return response
            result = response.json()
            agent_response = AgentResponse(result)
            return agent_response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            # Log to Inspect if available
            if self._inspect_logger:
                self._inspect_logger.log_inter_agent_call(
                    agent=self.agent_identifier,
                    message=message[:500] if message else "",  # Truncate for logging
                    response=agent_response.text[:500] if agent_response else None,
                    duration=time.time() - start_time,
                    error=error_msg,
                )

    async def call_with_data(self, data: Any, timeout: float | None = None) -> AgentResponse:
        """
        Call the agent with structured data.

        Args:
            data: Structured data (dict, list, etc.) to send
            timeout: Optional timeout override (default: use client timeout)

        Returns:
            AgentResponse with text, data, parts, and raw_response

        Raises:
            httpx.HTTPError: If the request fails after all retries
            ValueError: If agent identifier cannot be resolved

        Example:
            response = await client.call_with_data({"query": "test", "limit": 10})
            print(response.data)  # Convenient data access
            print(response.parts)  # All response parts
        """
        start_time = time.time()
        error_msg = None
        agent_response: AgentResponse | None = None

        try:
            # Build A2A message with DataPart
            data_part = DataPart(data=data)
            a2a_message = Message(
                message_id=str(uuid.uuid4()), role=Role.user, parts=[Part(root=data_part)]
            )

            # Build request
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            request_timeout = timeout if timeout is not None else self.timeout

            # Call with retry
            response = await self._call_with_retry(
                "POST",
                f"{self.target_url}/a2a/v1/messages",
                json=a2a_message.model_dump(by_alias=True),
                headers=headers,
                timeout=request_timeout,
            )

            # Parse and return response
            result = response.json()
            agent_response = AgentResponse(result)
            return agent_response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            # Log to Inspect if available
            if self._inspect_logger:
                # Truncate data representation for logging
                data_str = str(data)[:500] if data else ""
                self._inspect_logger.log_inter_agent_call(
                    agent=self.agent_identifier,
                    message=f"[DATA] {data_str}",
                    response=agent_response.text[:500] if agent_response else None,
                    duration=time.time() - start_time,
                    error=error_msg,
                )

    async def close(self) -> None:
        """
        Close the client.

        Currently a no-op (httpx clients are closed per request),
        but provided for API compatibility and future connection pooling.
        """
        pass
