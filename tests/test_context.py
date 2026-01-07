"""Tests for context module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from health_universe_a2a.context import (
    BaseContext,
    BackgroundContext,
    AgentContext,
    _SSEContext,
    StreamingContext,
)
from health_universe_a2a.types.extensions import UpdateImportance
from health_universe_a2a.update_client import BackgroundUpdateClient


class MockBackgroundUpdateClient(BackgroundUpdateClient):
    """Mock update client that inherits from real class to pass pydantic validation."""

    def __init__(self):
        # Don't call super().__init__() to avoid actual HTTP client creation
        self.job_id = "mock-job"
        self.api_key = "mock-key"
        self.base_url = "http://mock"
        self.client = MagicMock()
        self._calls: list[dict] = []

    async def post_update(self, *args, **kwargs) -> None:
        self._calls.append({"method": "post_update", "args": args, "kwargs": kwargs})

    async def close(self) -> None:
        pass


class TestBaseContext:
    """Tests for BaseContext class."""

    def test_initialization_with_defaults(self):
        """Test BaseContext with default values."""
        context = BaseContext()

        assert context.user_id is None
        assert context.thread_id is None
        assert context.file_access_token is None
        assert context.auth_token is None
        assert context.metadata == {}
        assert context.extensions is None

    def test_initialization_with_values(self):
        """Test BaseContext with provided values."""
        context = BaseContext(
            user_id="user-123",
            thread_id="thread-456",
            file_access_token="file-token",
            auth_token="jwt-token",
            metadata={"key": "value"},
            extensions=["ext1", "ext2"],
        )

        assert context.user_id == "user-123"
        assert context.thread_id == "thread-456"
        assert context.file_access_token == "file-token"
        assert context.auth_token == "jwt-token"
        assert context.metadata == {"key": "value"}
        assert context.extensions == ["ext1", "ext2"]

    def test_is_cancelled_returns_false(self):
        """Test is_cancelled always returns False (not implemented)."""
        context = BaseContext()
        assert context.is_cancelled() is False

        # Even with manual flag set (which shouldn't happen in normal use)
        context._cancelled = True
        assert context.is_cancelled() is True

    def test_create_inter_agent_client(self):
        """Test create_inter_agent_client creates properly configured client."""
        context = BaseContext(auth_token="jwt-token-123")

        with patch(
            "health_universe_a2a.inter_agent.InterAgentClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            client = context.create_inter_agent_client(
                agent_identifier="/local-agent",
                timeout=60.0,
                max_retries=5,
            )

            mock_client_cls.assert_called_once_with(
                agent_identifier="/local-agent",
                auth_token="jwt-token-123",
                local_base_url="http://localhost:8501",
                timeout=60.0,
                max_retries=5,
            )
            assert client == mock_client

    def test_create_inter_agent_client_with_env_url(self):
        """Test create_inter_agent_client uses LOCAL_AGENT_BASE_URL env var."""
        context = BaseContext()

        with patch.dict(
            "os.environ", {"LOCAL_AGENT_BASE_URL": "http://agents:9000"}
        ):
            with patch(
                "health_universe_a2a.inter_agent.InterAgentClient"
            ) as mock_client_cls:
                context.create_inter_agent_client("/agent")

                call_kwargs = mock_client_cls.call_args[1]
                assert call_kwargs["local_base_url"] == "http://agents:9000"

    def test_create_inter_agent_client_with_custom_base_url(self):
        """Test create_inter_agent_client with custom local_base_url."""
        context = BaseContext()

        with patch(
            "health_universe_a2a.inter_agent.InterAgentClient"
        ) as mock_client_cls:
            context.create_inter_agent_client(
                "/agent",
                local_base_url="http://custom:8080",
            )

            call_kwargs = mock_client_cls.call_args[1]
            assert call_kwargs["local_base_url"] == "http://custom:8080"


class TestBackgroundContext:
    """Tests for BackgroundContext class."""

    @pytest.fixture
    def mock_update_client(self):
        """Create mock update client."""
        return MockBackgroundUpdateClient()

    @pytest.fixture
    def context(self, mock_update_client):
        """Create BackgroundContext for testing."""
        return BackgroundContext(
            update_client=mock_update_client,
            job_id="job-123",
            user_id="user-456",
            thread_id="thread-789",
            file_access_token="file-token",
            auth_token="auth-token",
            metadata={"key": "value"},
        )

    def test_initialization(self, mock_update_client):
        """Test BackgroundContext initialization."""
        context = BackgroundContext(
            update_client=mock_update_client,
            job_id="job-123",
        )

        assert context.update_client == mock_update_client
        assert context.job_id == "job-123"
        assert context.loop is None

    def test_initialization_with_all_fields(self, mock_update_client):
        """Test BackgroundContext with all fields."""
        loop = asyncio.new_event_loop()

        context = BackgroundContext(
            update_client=mock_update_client,
            job_id="job-123",
            loop=loop,
            user_id="user-456",
            thread_id="thread-789",
            file_access_token="file-token",
            auth_token="auth-token",
            metadata={"key": "value"},
            extensions=["ext1"],
        )

        assert context.job_id == "job-123"
        assert context.loop == loop
        assert context.user_id == "user-456"
        assert context.thread_id == "thread-789"
        assert context.file_access_token == "file-token"
        assert context.auth_token == "auth-token"
        assert context.metadata == {"key": "value"}
        assert context.extensions == ["ext1"]

        loop.close()

    def test_document_client_property_lazy_init(self, context):
        """Test document_client is lazily initialized."""
        assert context._documents is None

        with patch(
            "health_universe_a2a.documents.DocumentClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # First access
            client1 = context.document_client
            assert mock_client_cls.call_count == 1
            assert client1 == mock_client

            # Second access - should return cached instance
            client2 = context.document_client
            assert mock_client_cls.call_count == 1  # Not called again
            assert client2 == mock_client

    def test_document_client_uses_env_url(self, mock_update_client):
        """Test document_client uses HU_NESTJS_URL env var."""
        context = BackgroundContext(
            update_client=mock_update_client,
            job_id="job-123",
            thread_id="thread-789",
            file_access_token="file-token",
        )

        with patch.dict(
            "os.environ", {"HU_NESTJS_URL": "https://custom.api.com"}
        ):
            with patch(
                "health_universe_a2a.documents.DocumentClient"
            ) as mock_client_cls:
                _ = context.document_client

                mock_client_cls.assert_called_once_with(
                    base_url="https://custom.api.com",
                    access_token="file-token",
                    thread_id="thread-789",
                )

    @pytest.mark.asyncio
    async def test_update_progress(self, context, mock_update_client):
        """Test update_progress calls update_client.post_update."""
        await context.update_progress(
            message="Processing...",
            progress=0.5,
            status="working",
            importance=UpdateImportance.NOTICE,
        )

        assert len(mock_update_client._calls) == 1
        call = mock_update_client._calls[0]
        assert call["kwargs"]["update_type"] == "progress"
        assert call["kwargs"]["status_message"] == "Processing..."
        assert call["kwargs"]["progress"] == 0.5
        assert call["kwargs"]["task_status"] == "working"
        assert call["kwargs"]["importance"] == UpdateImportance.NOTICE

    @pytest.mark.asyncio
    async def test_update_progress_defaults(self, context, mock_update_client):
        """Test update_progress with default values."""
        await context.update_progress("Working...")

        call = mock_update_client._calls[0]
        assert call["kwargs"]["progress"] is None
        assert call["kwargs"]["task_status"] == "working"
        assert call["kwargs"]["importance"] == UpdateImportance.INFO

    def test_update_progress_sync_no_loop(self, context):
        """Test update_progress_sync without event loop logs warning."""
        assert context.loop is None

        # Should not raise, just log warning
        context.update_progress_sync("Processing...")

    @pytest.mark.asyncio
    async def test_add_artifact(self, context, mock_update_client):
        """Test add_artifact calls update_client.post_update."""
        await context.add_artifact(
            name="Results",
            content='{"score": 0.95}',
            data_type="application/json",
            description="Analysis results",
            metadata={"format": "json"},
        )

        assert len(mock_update_client._calls) == 1
        call = mock_update_client._calls[0]
        assert call["kwargs"]["update_type"] == "artifact"
        artifact_data = call["kwargs"]["artifact_data"]
        assert artifact_data["name"] == "Results"
        assert artifact_data["content"] == '{"score": 0.95}'
        assert artifact_data["data_type"] == "application/json"
        assert artifact_data["description"] == "Analysis results"
        assert artifact_data["metadata"] == {"format": "json"}

    @pytest.mark.asyncio
    async def test_add_artifact_defaults(self, context, mock_update_client):
        """Test add_artifact with default values."""
        await context.add_artifact(
            name="Output",
            content="Hello world",
        )

        call = mock_update_client._calls[0]
        artifact_data = call["kwargs"]["artifact_data"]
        assert artifact_data["data_type"] == "text/plain"
        assert artifact_data["description"] == ""
        assert "metadata" not in artifact_data

    def test_add_artifact_sync_no_loop(self, context):
        """Test add_artifact_sync without event loop logs warning."""
        assert context.loop is None

        # Should not raise, just log warning
        context.add_artifact_sync("Output", "Hello")

    def test_inherits_base_context(self, context):
        """Test BackgroundContext inherits BaseContext methods."""
        # Should have is_cancelled
        assert hasattr(context, "is_cancelled")
        assert context.is_cancelled() is False

        # Should have create_inter_agent_client
        assert hasattr(context, "create_inter_agent_client")


class TestAgentContextAlias:
    """Tests for AgentContext alias."""

    def test_agent_context_is_background_context(self):
        """Test AgentContext is an alias for BackgroundContext."""
        assert AgentContext is BackgroundContext


class TestSSEContext:
    """Tests for internal _SSEContext class."""

    def test_sse_context_inherits_base_context(self):
        """Test _SSEContext inherits from BaseContext."""
        from health_universe_a2a.context import BaseContext

        assert issubclass(_SSEContext, BaseContext)

    def test_sse_context_has_required_fields(self):
        """Test _SSEContext has updater and request_context fields."""
        # _SSEContext is for internal use only, just verify structure
        from pydantic.fields import FieldInfo

        assert "updater" in _SSEContext.model_fields
        assert "request_context" in _SSEContext.model_fields

    def test_streaming_context_alias(self):
        """Test StreamingContext is an alias for _SSEContext."""
        assert StreamingContext is _SSEContext


class TestBackgroundContextSyncMethods:
    """Tests for sync methods with event loop."""

    @pytest.fixture
    def mock_update_client(self):
        """Create mock update client."""
        return MockBackgroundUpdateClient()

    def test_update_progress_sync_calls_async_method(self, mock_update_client):
        """Test update_progress_sync attempts to call the async method."""
        # Create a mock loop for testing
        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_loop.run_coroutine_threadsafe = MagicMock()

        # Patch the run_coroutine_threadsafe to capture the call
        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_run.return_value = mock_future

            context = BackgroundContext(
                update_client=mock_update_client,
                job_id="job-123",
                loop=mock_loop,
            )

            context.update_progress_sync(
                message="From thread",
                progress=0.5,
                importance=UpdateImportance.NOTICE,
            )

            # Verify run_coroutine_threadsafe was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            # First arg is a coroutine, second is the loop
            assert call_args[0][1] == mock_loop

    def test_add_artifact_sync_calls_async_method(self, mock_update_client):
        """Test add_artifact_sync attempts to call the async method."""
        mock_loop = MagicMock()
        mock_future = MagicMock()

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_run.return_value = mock_future

            context = BackgroundContext(
                update_client=mock_update_client,
                job_id="job-123",
                loop=mock_loop,
            )

            context.add_artifact_sync(
                name="Thread Result",
                content="Data from thread",
                data_type="text/plain",
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][1] == mock_loop

    def test_update_progress_sync_handles_timeout(self, mock_update_client):
        """Test update_progress_sync handles timeout gracefully."""
        import concurrent.futures

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_run.return_value = mock_future

            context = BackgroundContext(
                update_client=mock_update_client,
                job_id="job-123",
                loop=mock_loop,
            )

            # Should not raise
            context.update_progress_sync("Message")

    def test_add_artifact_sync_handles_exception(self, mock_update_client):
        """Test add_artifact_sync handles exceptions gracefully."""
        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Connection failed")

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_run.return_value = mock_future

            context = BackgroundContext(
                update_client=mock_update_client,
                job_id="job-123",
                loop=mock_loop,
            )

            # Should not raise
            context.add_artifact_sync("Output", "Content")


class TestContextIntegration:
    """Integration tests for context objects."""

    @pytest.fixture
    def mock_update_client(self):
        """Create mock update client."""
        return MockBackgroundUpdateClient()

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_update_client):
        """Test a typical agent workflow using context."""
        context = BackgroundContext(
            update_client=mock_update_client,
            job_id="job-123",
            thread_id="thread-456",
            file_access_token="file-token",
        )

        # Simulate agent workflow
        await context.update_progress("Starting analysis...", 0.1)
        await context.update_progress("Processing data...", 0.5)
        await context.add_artifact(
            name="Intermediate Results",
            content='{"partial": true}',
            data_type="application/json",
        )
        await context.update_progress("Finalizing...", 0.9)

        # Verify all calls were made
        assert len(mock_update_client._calls) == 4
        assert mock_update_client._calls[0]["kwargs"]["status_message"] == "Starting analysis..."
        assert mock_update_client._calls[1]["kwargs"]["status_message"] == "Processing data..."
        assert mock_update_client._calls[2]["kwargs"]["artifact_data"]["name"] == "Intermediate Results"
        assert mock_update_client._calls[3]["kwargs"]["status_message"] == "Finalizing..."
