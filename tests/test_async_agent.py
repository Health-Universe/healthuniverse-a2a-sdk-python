"""Tests for AsyncAgent class."""

import asyncio
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from health_universe_a2a.async_agent import AsyncAgent
from health_universe_a2a.context import BackgroundContext
from health_universe_a2a.types.validation import ValidationAccepted, ValidationRejected
from health_universe_a2a.update_client import BackgroundUpdateClient


class MockBackgroundUpdateClient(BackgroundUpdateClient):
    """Mock update client that inherits from real class to pass pydantic validation."""

    def __init__(self):
        # Don't call super().__init__() to avoid actual HTTP client creation
        self.job_id = "mock-job"
        self.api_key = "mock-key"
        self.base_url = "http://mock"
        self.client = MagicMock()
        self._post_completion_called = False
        self._post_failure_called = False
        self._failure_message: str | None = None

    async def post_update(self, *args, **kwargs) -> None:
        pass

    async def post_completion(self, message: str) -> None:
        self._post_completion_called = True

    async def post_failure(self, error: str) -> None:
        self._post_failure_called = True
        self._failure_message = error

    async def close(self) -> None:
        pass


class ConcreteAsyncAgent(AsyncAgent):
    """Concrete implementation of AsyncAgent for testing."""

    def __init__(self, process_result: str = "done", process_error: Exception | None = None):
        super().__init__()
        self._process_result = process_result
        self._process_error = process_error
        self.process_message_called = False
        self.received_context: BackgroundContext | None = None

    def get_agent_name(self) -> str:
        return "Test Async Agent"

    def get_agent_description(self) -> str:
        return "Agent for testing async functionality"

    async def process_message(self, message: str, context: BackgroundContext) -> str:
        self.process_message_called = True
        self.received_context = context
        if self._process_error:
            raise self._process_error
        return self._process_result


class AsyncAgentWithValidation(AsyncAgent):
    """AsyncAgent with custom validation."""

    def __init__(self, validation_result: ValidationAccepted | ValidationRejected):
        super().__init__()
        self._validation_result = validation_result

    def get_agent_name(self) -> str:
        return "Validating Async Agent"

    def get_agent_description(self) -> str:
        return "Agent with custom validation"

    async def validate_message(
        self, message: str, metadata: dict[str, Any]
    ) -> ValidationAccepted | ValidationRejected:
        return self._validation_result

    async def process_message(self, message: str, context: BackgroundContext) -> str:
        return "processed"


class AsyncAgentWithHooks(AsyncAgent):
    """AsyncAgent with lifecycle hooks."""

    def __init__(self):
        super().__init__()
        self.on_start_called = False
        self.on_complete_called = False
        self.on_error_called = False
        self.start_context: BackgroundContext | None = None
        self.complete_context: BackgroundContext | None = None

    def get_agent_name(self) -> str:
        return "Hook Agent"

    def get_agent_description(self) -> str:
        return "Agent with hooks"

    async def on_task_start(self, message: str, context: BackgroundContext) -> None:
        self.on_start_called = True
        self.start_context = context

    async def on_task_complete(
        self, message: str, result: str, context: BackgroundContext
    ) -> None:
        self.on_complete_called = True
        self.complete_context = context

    async def on_task_error(
        self, message: str, error: Exception, context: BackgroundContext
    ) -> None:
        self.on_error_called = True

    async def process_message(self, message: str, context: BackgroundContext) -> str:
        return "hook result"


class TestAsyncAgentBasics:
    """Test basic AsyncAgent functionality."""

    def test_instantiation(self):
        """Test AsyncAgent can be instantiated."""
        agent = ConcreteAsyncAgent()
        assert agent.get_agent_name() == "Test Async Agent"

    def test_get_max_duration_seconds(self):
        """Test default max duration."""
        agent = ConcreteAsyncAgent()
        # Default is 3600 (1 hour) for async agents
        assert agent.get_max_duration_seconds() >= 60

    def test_custom_max_duration(self):
        """Test custom max duration."""

        class CustomDurationAgent(ConcreteAsyncAgent):
            def get_max_duration_seconds(self) -> int:
                return 7200  # 2 hours

        agent = CustomDurationAgent()
        assert agent.get_max_duration_seconds() == 7200


class TestAsyncAgentValidation:
    """Test AsyncAgent validation."""

    @pytest.mark.asyncio
    async def test_default_validation_accepts(self):
        """Test default validation accepts messages."""
        agent = ConcreteAsyncAgent()
        result = await agent.validate_message("test message", {})
        assert isinstance(result, ValidationAccepted)

    @pytest.mark.asyncio
    async def test_custom_validation_accepts(self):
        """Test custom validation can accept."""
        agent = AsyncAgentWithValidation(
            ValidationAccepted(estimated_duration_seconds=120)
        )
        result = await agent.validate_message("test", {})
        assert isinstance(result, ValidationAccepted)
        assert result.estimated_duration_seconds == 120

    @pytest.mark.asyncio
    async def test_custom_validation_rejects(self):
        """Test custom validation can reject."""
        agent = AsyncAgentWithValidation(
            ValidationRejected(reason="Invalid input")
        )
        result = await agent.validate_message("bad input", {})
        assert isinstance(result, ValidationRejected)
        assert result.reason == "Invalid input"


class TestBackgroundWork:
    """Test _run_background_work method."""

    @pytest.fixture
    def mock_update_client(self):
        """Create mock BackgroundUpdateClient that passes pydantic validation."""
        mock_client = MockBackgroundUpdateClient()
        with patch(
            "health_universe_a2a.async_agent.BackgroundUpdateClient",
            return_value=mock_client,
        ):
            yield mock_client

    @pytest.mark.asyncio
    async def test_background_work_success(self, mock_update_client):
        """Test successful background work execution."""
        agent = ConcreteAsyncAgent(process_result="success!")

        await agent._run_background_work(
            message="test message",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
            user_id="user-1",
            thread_id="thread-1",
        )

        assert agent.process_message_called
        assert mock_update_client._post_completion_called

    @pytest.mark.asyncio
    async def test_background_work_failure(self, mock_update_client):
        """Test background work with error."""
        agent = ConcreteAsyncAgent(process_error=ValueError("Something went wrong"))

        await agent._run_background_work(
            message="test message",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
        )

        assert mock_update_client._post_failure_called

    @pytest.mark.asyncio
    async def test_auth_token_propagation(self, mock_update_client):
        """Test auth_token is passed to BackgroundContext."""
        agent = ConcreteAsyncAgent()

        await agent._run_background_work(
            message="test",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
            auth_token="jwt-token-123",
        )

        # Verify auth_token was passed to context
        assert agent.received_context is not None
        assert agent.received_context.auth_token == "jwt-token-123"

    @pytest.mark.asyncio
    async def test_context_has_all_fields(self, mock_update_client):
        """Test BackgroundContext receives all expected fields."""
        agent = ConcreteAsyncAgent()

        await agent._run_background_work(
            message="test",
            job_id="job-123",
            api_key="api-key",
            metadata={"key": "value"},
            task_id="task-456",
            context_id="ctx-789",
            user_id="user-abc",
            thread_id="thread-xyz",
            file_access_token="file-token",
            auth_token="auth-token",
            extensions=["ext1", "ext2"],
        )

        ctx = agent.received_context
        assert ctx is not None
        assert ctx.user_id == "user-abc"
        assert ctx.thread_id == "thread-xyz"
        assert ctx.file_access_token == "file-token"
        assert ctx.auth_token == "auth-token"
        assert ctx.metadata == {"key": "value"}
        assert ctx.extensions == ["ext1", "ext2"]
        assert ctx.job_id == "job-123"

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_called(self, mock_update_client):
        """Test lifecycle hooks are called."""
        agent = AsyncAgentWithHooks()

        await agent._run_background_work(
            message="test",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
        )

        assert agent.on_start_called
        assert agent.on_complete_called
        assert not agent.on_error_called

    @pytest.mark.asyncio
    async def test_error_hook_called_on_failure(self, mock_update_client):
        """Test on_task_error hook is called on failure."""

        class FailingHookAgent(AsyncAgentWithHooks):
            async def process_message(self, message: str, context: BackgroundContext) -> str:
                raise RuntimeError("Processing failed")

        agent = FailingHookAgent()

        await agent._run_background_work(
            message="test",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
        )

        assert agent.on_start_called
        assert not agent.on_complete_called
        assert agent.on_error_called


class TestAsyncAgentTimeout:
    """Test timeout handling."""

    @pytest.fixture
    def mock_update_client(self):
        """Create mock BackgroundUpdateClient that passes pydantic validation."""
        mock_client = MockBackgroundUpdateClient()
        with patch(
            "health_universe_a2a.async_agent.BackgroundUpdateClient",
            return_value=mock_client,
        ):
            yield mock_client

    @pytest.mark.asyncio
    async def test_timeout_triggers_failure(self, mock_update_client):
        """Test that timeout triggers failure handling."""

        class SlowAgent(AsyncAgent):
            def get_agent_name(self) -> str:
                return "Slow Agent"

            def get_agent_description(self) -> str:
                return "Takes too long"

            def get_max_duration_seconds(self) -> int:
                return 1  # 1 second timeout

            async def process_message(self, message: str, context: BackgroundContext) -> str:
                await asyncio.sleep(10)  # Sleep longer than timeout
                return "done"

        agent = SlowAgent()

        await agent._run_background_work(
            message="test",
            job_id="job-123",
            api_key="api-key",
            metadata={},
            task_id="task-456",
            context_id="ctx-789",
        )

        # Should have called post_failure due to timeout
        assert mock_update_client._post_failure_called
        assert mock_update_client._failure_message is not None
        assert "exceeded maximum duration" in mock_update_client._failure_message.lower()
