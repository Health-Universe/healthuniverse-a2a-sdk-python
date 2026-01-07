"""Tests for base A2AAgent class"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from health_universe_a2a import (
    Agent,
    AgentContext,
    ValidationAccepted,
    ValidationRejected,
)
from health_universe_a2a.types.validation import ValidationResult
from health_universe_a2a.update_client import BackgroundUpdateClient


# Helper function to create a test context with mocks
def create_test_context() -> AgentContext:
    """Create an AgentContext with mock update_client for testing."""
    update_client = MagicMock(spec=BackgroundUpdateClient)
    update_client.post_update = AsyncMock()
    update_client.post_completion = AsyncMock()
    update_client.post_failure = AsyncMock()
    update_client.close = AsyncMock()

    # Use model_construct to bypass Pydantic validation for tests
    return AgentContext.model_construct(
        update_client=update_client,
        job_id="test-job-123",
        user_id="test-user",
        thread_id="test-thread",
        file_access_token="test-token",
    )


class TestAgent(Agent):
    """Simple test agent for testing base functionality."""

    def get_agent_name(self) -> str:
        return "Test Agent"

    def get_agent_description(self) -> str:
        return "An agent for testing"

    async def process_message(self, message: str, context: AgentContext) -> str:
        return f"Processed: {message}"


class TestA2AAgentBase:
    """Tests for A2AAgentBase (via Agent)."""

    def test_agent_initialization(self) -> None:
        """Agent should initialize without errors."""
        agent = TestAgent()
        assert agent.get_agent_name() == "Test Agent"
        assert agent.get_agent_description() == "An agent for testing"

    def test_default_version(self) -> None:
        """Agent should have default version 1.0.0."""
        agent = TestAgent()
        assert agent.get_agent_version() == "1.0.0"

    def test_default_supported_formats(self) -> None:
        """Agent should support text/plain and application/json by default."""
        agent = TestAgent()
        input_formats = agent.get_supported_input_formats()
        output_formats = agent.get_supported_output_formats()

        assert "text/plain" in input_formats
        assert "application/json" in input_formats
        assert "text/plain" in output_formats
        assert "application/json" in output_formats

    @pytest.mark.asyncio
    async def test_default_validation_accepts_all(self) -> None:
        """Default validation should accept all messages."""
        agent = TestAgent()
        result = await agent.validate_message("test message", {})

        assert result.status == "accepted"
        assert result.estimated_duration_seconds is None

    @pytest.mark.asyncio
    async def test_process_message(self) -> None:
        """Agent should process messages correctly."""
        agent = TestAgent()
        context = create_test_context()
        result = await agent.process_message("hello", context)

        assert result == "Processed: hello"

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_are_optional(self) -> None:
        """Lifecycle hooks should be no-ops by default."""
        agent = TestAgent()
        context = create_test_context()

        # Should not raise any errors
        await agent.on_startup()
        await agent.on_shutdown()
        await agent.on_task_start("message", context)
        await agent.on_task_complete("message", "result", context)
        error_msg = await agent.on_task_error("message", ValueError(), context)

        assert error_msg is None  # Default returns None


class TestAgentWithCustomValidation(Agent):
    """Test agent with custom validation."""

    def get_agent_name(self) -> str:
        return "Validation Test Agent"

    def get_agent_description(self) -> str:
        return "Tests custom validation"

    async def validate_message(self, message: str, metadata: dict[str, Any]) -> ValidationResult:
        if len(message) < 5:
            return ValidationRejected(reason="Message too short (min 5 chars)")

        return ValidationAccepted(estimated_duration_seconds=10)

    async def process_message(self, message: str, context: AgentContext) -> str:
        return "processed"


class TestCustomValidation:
    """Tests for custom validation."""

    @pytest.mark.asyncio
    async def test_validation_rejects_short_messages(self) -> None:
        """Validation should reject messages that are too short."""
        agent = TestAgentWithCustomValidation()
        result = await agent.validate_message("hi", {})

        assert result.status == "rejected"
        assert "too short" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_validation_accepts_long_messages(self) -> None:
        """Validation should accept messages that meet requirements."""
        agent = TestAgentWithCustomValidation()
        result = await agent.validate_message("hello world", {})

        assert result.status == "accepted"
        assert result.estimated_duration_seconds == 10


# Test agent with lifecycle hooks for testing
class TestAgentWithHooks(Agent):
    """Test agent that tracks lifecycle hook calls."""

    def __init__(self) -> None:
        super().__init__()
        self.hook_calls: list[str] = []

    def get_agent_name(self) -> str:
        return "Hooks Test Agent"

    def get_agent_description(self) -> str:
        return "Tests lifecycle hooks"

    async def process_message(self, message: str, context: AgentContext) -> str:
        self.hook_calls.append("process_message")
        return f"Processed: {message}"

    async def on_task_start(self, message: str, context: AgentContext) -> None:
        self.hook_calls.append("on_task_start")

    async def on_task_complete(self, message: str, result: str, context: AgentContext) -> None:
        self.hook_calls.append("on_task_complete")

    async def on_task_error(
        self, message: str, error: Exception, context: AgentContext
    ) -> str | None:
        self.hook_calls.append("on_task_error")
        return f"Custom error: {str(error)}"


class TestAgentWithCustomConfig(Agent):
    """Test agent with custom configuration."""

    def get_agent_name(self) -> str:
        return "Custom Config Agent"

    def get_agent_description(self) -> str:
        return "Has custom configuration"

    def get_agent_version(self) -> str:
        return "2.5.0"

    def get_supported_input_formats(self) -> list[str]:
        return ["text/plain", "text/csv"]

    def get_supported_output_formats(self) -> list[str]:
        return ["application/json", "text/html"]

    def get_provider_organization(self) -> str:
        return "Test Organization"

    def get_provider_url(self) -> str:
        return "https://test.example.com"

    async def process_message(self, message: str, context: AgentContext) -> str:
        return "processed"


class TestAgentCardCreation:
    """Tests for AgentCard creation."""

    def test_agent_card_basic_fields(self) -> None:
        """AgentCard should include basic required fields."""
        agent = TestAgent()
        card = agent.create_agent_card()

        assert card.protocol_version == "0.3.0"
        assert card.name == "Test Agent"
        assert card.description == "An agent for testing"
        assert card.version == "1.0.0"
        assert card.preferred_transport == "JSONRPC"

    def test_agent_card_custom_version(self) -> None:
        """AgentCard should use custom version from agent."""
        agent = TestAgentWithCustomConfig()
        card = agent.create_agent_card()

        assert card.version == "2.5.0"

    def test_agent_card_provider_info(self) -> None:
        """AgentCard should include provider information."""
        agent = TestAgentWithCustomConfig()
        card = agent.create_agent_card()

        assert card.provider is not None
        assert card.provider.organization == "Test Organization"
        assert card.provider.url == "https://test.example.com"

    def test_agent_card_input_output_formats(self) -> None:
        """AgentCard should include custom input/output formats."""
        agent = TestAgentWithCustomConfig()
        card = agent.create_agent_card()

        assert card.default_input_modes == ["text/plain", "text/csv"]
        assert card.default_output_modes == ["application/json", "text/html"]

    def test_agent_card_default_formats(self) -> None:
        """AgentCard should have sensible default formats."""
        agent = TestAgent()
        card = agent.create_agent_card()

        assert "text/plain" in card.default_input_modes
        assert "application/json" in card.default_input_modes
        assert "text/plain" in card.default_output_modes
        assert "application/json" in card.default_output_modes

    def test_agent_card_capabilities(self) -> None:
        """AgentCard should include capabilities."""
        agent = TestAgent()
        card = agent.create_agent_card()

        assert card.capabilities is not None
        # Agent (AsyncAgent) does NOT support streaming, supports push notifications
        assert card.capabilities.streaming is False
        assert card.capabilities.push_notifications is True

    def test_agent_card_url_configuration(self) -> None:
        """AgentCard should include URL from agent configuration."""
        agent = TestAgent()
        card = agent.create_agent_card()

        # Should have a URL (from environment or default)
        assert card.url is not None
        assert card.additional_interfaces is not None
        assert len(card.additional_interfaces) > 0
        assert card.additional_interfaces[0].transport == "JSONRPC"

    def test_agent_card_extensions(self) -> None:
        """AgentCard should include auto-configured extensions."""
        agent = TestAgent()
        card = agent.create_agent_card()

        # Agent (AsyncAgent) automatically configures background job and file access extensions
        assert card.capabilities is not None
        extensions = card.capabilities.extensions
        assert extensions is not None
        assert len(extensions) >= 2  # background_job and file_access_v2


class TestContextIntegration:
    """Tests for context integration."""

    @pytest.mark.asyncio
    async def test_process_message_can_use_context(self) -> None:
        """Agent should be able to use context during processing."""

        class ContextUsingAgent(Agent):
            def get_agent_name(self) -> str:
                return "Context Agent"

            def get_agent_description(self) -> str:
                return "Uses context"

            async def process_message(self, message: str, context: AgentContext) -> str:
                # Verify context is passed and accessible
                assert context is not None
                # Test accessing context attributes
                _ = context.user_id
                _ = context.metadata
                _ = context.job_id
                return "Done!"

        agent = ContextUsingAgent()
        context = create_test_context()

        result = await agent.process_message("test", context)

        assert result == "Done!"

    @pytest.mark.asyncio
    async def test_context_update_progress(self) -> None:
        """Context should allow progress updates."""
        context = create_test_context()

        await context.update_progress("Working...", 0.5)

        context.update_client.post_update.assert_called_once()  # type: ignore[union-attr]
