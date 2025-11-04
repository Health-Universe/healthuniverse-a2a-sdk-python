"""Tests for base A2AAgent class"""

from unittest.mock import AsyncMock

import pytest

from health_universe_a2a import (
    A2AAgent,
    MessageContext,
    ValidationAccepted,
    ValidationRejected,
)
from health_universe_a2a.types.validation import ValidationResult


class TestAgent(A2AAgent[MessageContext]):
    """Simple test agent for testing base functionality."""

    def get_agent_name(self) -> str:
        return "Test Agent"

    def get_agent_description(self) -> str:
        return "An agent for testing"

    async def process_message(self, message: str, context: MessageContext) -> str:
        return f"Processed: {message}"


class TestA2AAgent:
    """Tests for A2AAgent base class."""

    def test_agent_initialization(self) -> None:
        """Agent should initialize without errors."""
        agent = TestAgent()
        assert agent.get_agent_name() == "Test Agent"
        assert agent.get_agent_description() == "An agent for testing"

    def test_default_version(self) -> None:
        """Agent should have default version 1.0.0."""
        agent = TestAgent()
        assert agent.get_agent_version() == "1.0.0"

    def test_default_file_access(self) -> None:
        """Agent should not require file access by default."""
        agent = TestAgent()
        assert agent.requires_file_access() is False

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
        context = MessageContext()
        result = await agent.process_message("hello", context)

        assert result == "Processed: hello"

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_are_optional(self) -> None:
        """Lifecycle hooks should be no-ops by default."""
        agent = TestAgent()
        context = MessageContext()

        # Should not raise any errors
        await agent.on_startup()
        await agent.on_shutdown()
        await agent.on_task_start("message", context)
        await agent.on_task_complete("message", "result", context)
        error_msg = await agent.on_task_error("message", ValueError(), context)

        assert error_msg is None  # Default returns None


class TestAgentWithCustomValidation(A2AAgent[MessageContext]):
    """Test agent with custom validation."""

    def get_agent_name(self) -> str:
        return "Validation Test Agent"

    def get_agent_description(self) -> str:
        return "Tests custom validation"

    async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
        if len(message) < 5:
            return ValidationRejected(reason="Message too short (min 5 chars)")

        return ValidationAccepted(estimated_duration_seconds=10)

    async def process_message(self, message: str, context: MessageContext) -> str:
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


class TestAgentWithFileAccess(A2AAgent[MessageContext]):
    """Test agent that requires file access."""

    def get_agent_name(self) -> str:
        return "File Access Agent"

    def get_agent_description(self) -> str:
        return "Needs file access"

    def requires_file_access(self) -> bool:
        return True

    async def process_message(self, message: str, context: MessageContext) -> str:
        return "processed"


class TestFileAccessConfiguration:
    """Tests for file access configuration."""

    def test_file_access_can_be_enabled(self) -> None:
        """Agent should be able to require file access."""
        agent = TestAgentWithFileAccess()
        assert agent.requires_file_access() is True


# Test agent with lifecycle hooks for testing
class TestAgentWithHooks(A2AAgent[MessageContext]):
    """Test agent that tracks lifecycle hook calls."""

    def __init__(self) -> None:
        super().__init__()
        self.hook_calls: list[str] = []

    def get_agent_name(self) -> str:
        return "Hooks Test Agent"

    def get_agent_description(self) -> str:
        return "Tests lifecycle hooks"

    async def process_message(self, message: str, context: MessageContext) -> str:
        self.hook_calls.append("process_message")
        return f"Processed: {message}"

    async def on_task_start(self, message: str, context: MessageContext) -> None:
        self.hook_calls.append("on_task_start")

    async def on_task_complete(self, message: str, result: str, context: MessageContext) -> None:
        self.hook_calls.append("on_task_complete")

    async def on_task_error(
        self, message: str, error: Exception, context: MessageContext
    ) -> str | None:
        self.hook_calls.append("on_task_error")
        return f"Custom error: {str(error)}"


class TestAgentThatErrors(A2AAgent[MessageContext]):
    """Test agent that raises an error during processing."""

    def get_agent_name(self) -> str:
        return "Error Agent"

    def get_agent_description(self) -> str:
        return "Always errors"

    async def process_message(self, message: str, context: MessageContext) -> str:
        raise ValueError("Processing failed!")


class TestHandleRequestFlow:
    """Tests for handle_request() orchestration."""

    @pytest.mark.asyncio
    async def test_handle_request_with_rejection(self) -> None:
        """handle_request should handle validation rejection properly."""
        agent = TestAgentWithCustomValidation()
        context = MessageContext()

        # Mock the updater - return immediately to avoid Message creation
        mock_updater = AsyncMock()
        context._updater = mock_updater

        # Short message should be rejected
        result = await agent.handle_request("hi", context, {})

        # Should return None for rejection
        assert result is None

        # Should have called reject on the updater
        mock_updater.reject.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_request_with_acceptance(self) -> None:
        """handle_request should process accepted messages."""
        agent = TestAgentWithCustomValidation()
        context = MessageContext()
        context._updater = AsyncMock()

        # Long message should be accepted and processed
        result = await agent.handle_request("hello world", context, {})

        # Should return the processed result
        assert result == "processed"

        # Should NOT have called reject
        context._updater.reject.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_request_calls_lifecycle_hooks_in_order(self) -> None:
        """handle_request should call lifecycle hooks in correct order."""
        agent = TestAgentWithHooks()
        context = MessageContext()

        result = await agent.handle_request("test message", context, {})

        # Verify result
        assert result == "Processed: test message"

        # Verify hooks were called in order
        assert agent.hook_calls == [
            "on_task_start",
            "process_message",
            "on_task_complete",
        ]

    @pytest.mark.asyncio
    async def test_handle_request_with_processing_error(self) -> None:
        """handle_request should handle processing errors."""
        agent = TestAgentThatErrors()
        context = MessageContext()
        context._updater = AsyncMock()

        # Should raise the error
        with pytest.raises(ValueError, match="Processing failed!"):
            await agent.handle_request("test", context, {})

        # Should have sent error update
        context._updater.update_progress.assert_called_once()
        call_args = context._updater.update_progress.call_args
        assert "Processing failed!" in call_args.kwargs["message"]
        assert call_args.kwargs["status"] == "error"

    @pytest.mark.asyncio
    async def test_handle_request_calls_error_hook(self) -> None:
        """handle_request should call on_task_error when processing fails."""
        agent = TestAgentWithHooks()
        context = MessageContext()
        context._updater = AsyncMock()

        # Make process_message raise an error
        _ = agent.process_message

        async def error_process(message: str, context: MessageContext) -> str:
            agent.hook_calls.append("process_message")
            raise RuntimeError("Test error")

        agent.process_message = error_process # type: ignore

        # Should raise the error
        with pytest.raises(RuntimeError, match="Test error"):
            await agent.handle_request("test", context, {})

        # Verify error hook was called
        assert "on_task_error" in agent.hook_calls

        # Verify hooks were called in order
        assert agent.hook_calls == [
            "on_task_start",
            "process_message",
            "on_task_error",
        ]

    @pytest.mark.asyncio
    async def test_handle_request_uses_custom_error_message(self) -> None:
        """handle_request should use custom error message from on_task_error."""
        agent = TestAgentWithHooks()
        context = MessageContext()
        context._updater = AsyncMock()

        # Make process_message raise an error
        async def error_process(message: str, context: MessageContext) -> str:
            agent.hook_calls.append("process_message")
            raise ValueError("Internal error")

        agent.process_message = error_process

        # Should raise the error
        with pytest.raises(ValueError):
            await agent.handle_request("test", context, {})

        # Should have sent custom error message
        context._updater.update_progress.assert_called_once()
        call_args = context._updater.update_progress.call_args
        assert call_args.kwargs["message"] == "Custom error: Internal error"


class TestAgentWithCustomConfig(A2AAgent[MessageContext]):
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

    def requires_file_access(self) -> bool:
        return True

    async def process_message(self, message: str, context: MessageContext) -> str:
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
        assert card.capabilities.streaming is False
        assert card.capabilities.push_notifications is False

    def test_agent_card_url_configuration(self) -> None:
        """AgentCard should include URL from agent configuration."""
        agent = TestAgent()
        card = agent.create_agent_card()

        # Should have a URL (from environment or default)
        assert card.url is not None
        assert card.additional_interfaces is not None
        assert len(card.additional_interfaces) > 0
        assert card.additional_interfaces[0].transport == "JSONRPC"


class TestContextIntegration:
    """Tests for context integration."""

    @pytest.mark.asyncio
    async def test_process_message_can_use_context(self) -> None:
        """Agent should be able to use context during processing."""

        class ContextUsingAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "Context Agent"

            def get_agent_description(self) -> str:
                return "Uses context"

            async def process_message(self, message: str, context: MessageContext) -> str:
                # Verify context is passed and accessible
                assert context is not None
                # Test accessing context attributes
                _ = context.user_id
                _ = context.metadata
                return "Done!"

        agent = ContextUsingAgent()
        context = MessageContext()

        result = await agent.handle_request("test", context, {})

        assert result == "Done!"
