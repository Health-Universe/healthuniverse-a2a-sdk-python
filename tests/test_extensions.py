"""Tests for types/extensions module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a.types import TaskState

from health_universe_a2a.types.extensions import (
    UpdateImportance,
    FileAccessExtensionContext,
    FileAccessExtensionParams,
    BackgroundJobExtensionParams,
    BackgroundJobExtensionResponse,
    BackgroundTaskResults,
    ack_background_job_enqueued,
    notify_on_task_completion,
    BACKGROUND_JOB_EXTENSION_URI,
)


class TestUpdateImportance:
    """Tests for UpdateImportance enum."""

    def test_values(self):
        """Test enum values."""
        assert UpdateImportance.ERROR.value == "error"
        assert UpdateImportance.NOTICE.value == "notice"
        assert UpdateImportance.INFO.value == "info"
        assert UpdateImportance.DEBUG.value == "debug"


class TestFileAccessExtensionParams:
    """Tests for FileAccessExtensionParams."""

    def test_model_validate(self):
        """Test creating params from dict."""
        data = {
            "access_token": "jwt-token",
            "context": {
                "user_id": "user-123",
                "thread_id": "thread-456",
            },
        }

        params = FileAccessExtensionParams.model_validate(data)

        assert params.access_token.get_secret_value() == "jwt-token"
        assert params.context.user_id == "user-123"
        assert params.context.thread_id == "thread-456"


class TestBackgroundJobExtensionParams:
    """Tests for BackgroundJobExtensionParams."""

    def test_model_validate(self):
        """Test creating params from dict."""
        data = {
            "api_key": "api-key-123",
            "job_id": "job-456",
        }

        params = BackgroundJobExtensionParams.model_validate(data)

        assert params.api_key == "api-key-123"
        assert params.job_id == "job-456"


class TestBackgroundTaskResults:
    """Tests for BackgroundTaskResults."""

    def test_model_creation(self):
        """Test creating results model."""
        from a2a.types import Artifact, Part, TextPart

        artifact = Artifact(
            artifact_id="art-1",
            parts=[Part(root=TextPart(text="Result"))],
        )

        results = BackgroundTaskResults(
            job_id="job-123",
            state=TaskState.completed,
            artifacts=[artifact],
        )

        assert results.job_id == "job-123"
        assert results.state == TaskState.completed
        assert len(results.artifacts) == 1


class TestAckBackgroundJobEnqueued:
    """Tests for ack_background_job_enqueued helper."""

    def test_creates_message_with_defaults(self):
        """Test creating ack message with default content."""
        message = ack_background_job_enqueued(job_id="job-123")

        assert message.message_id  # Should have a UUID
        assert len(message.parts) == 1
        assert "background" in message.parts[0].root.text.lower()
        assert BACKGROUND_JOB_EXTENSION_URI in message.extensions
        assert message.metadata[BACKGROUND_JOB_EXTENSION_URI]["job_id"] == "job-123"

    def test_creates_message_with_custom_content(self):
        """Test creating ack message with custom content."""
        message = ack_background_job_enqueued(
            job_id="job-123",
            content="Starting analysis...",
        )

        assert message.parts[0].root.text == "Starting analysis..."

    def test_creates_message_with_task_and_context_ids(self):
        """Test creating ack message with task and context IDs."""
        message = ack_background_job_enqueued(
            job_id="job-123",
            task_id="task-456",
            context_id="ctx-789",
        )

        assert message.task_id == "task-456"
        assert message.context_id == "ctx-789"


class TestNotifyOnTaskCompletion:
    """Tests for notify_on_task_completion function."""

    @pytest.fixture
    def task_results(self):
        """Create test task results."""
        return BackgroundTaskResults(
            job_id="job-123",
            state=TaskState.completed,
            artifacts=[],
        )

    @pytest.mark.asyncio
    async def test_skips_notification_in_local_mode(self, task_results):
        """Test notification is skipped for local-mode-key."""
        with patch("httpx.AsyncClient") as mock_client:
            await notify_on_task_completion(
                api_key="local-mode-key",
                result=task_results,
            )

            # Should not have created any HTTP client
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_notification_success(self, task_results):
        """Test successful notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock()

            await notify_on_task_completion(
                api_key="api-key-123",
                result=task_results,
                url="https://webhook.example.com",
            )

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://webhook.example.com"
            assert call_args[1]["headers"]["X-Api-Key"] == "api-key-123"

    @pytest.mark.asyncio
    async def test_sends_notification_with_default_url(self, task_results):
        """Test notification uses default URL from env."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.dict("os.environ", {"CALLBACK_URL": "https://env-callback.com"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client_cls.return_value.__aexit__ = AsyncMock()

                await notify_on_task_completion(
                    api_key="api-key-123",
                    result=task_results,
                )

                call_args = mock_client.post.call_args
                assert call_args[0][0] == "https://env-callback.com"

    @pytest.mark.asyncio
    async def test_handles_non_200_response(self, task_results):
        """Test handling of non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock()

            # Should not raise
            await notify_on_task_completion(
                api_key="api-key-123",
                result=task_results,
                url="https://webhook.example.com",
            )

    @pytest.mark.asyncio
    async def test_handles_exception(self, task_results):
        """Test handling of HTTP exception."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock()

            # Should not raise
            await notify_on_task_completion(
                api_key="api-key-123",
                result=task_results,
                url="https://webhook.example.com",
            )
