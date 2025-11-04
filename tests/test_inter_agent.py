"""Tests for inter-agent communication functionality."""

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from health_universe_a2a.inter_agent import AgentResponse, InterAgentClient


class TestAgentResponse:
    """Tests for AgentResponse class."""

    def test_text_property_single_text_part(self) -> None:
        """Should extract text from single text part."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Hello World"}
                ]
            }
        }
        response = AgentResponse(raw)
        assert response.text == "Hello World"

    def test_text_property_multiple_text_parts(self) -> None:
        """Should concatenate multiple text parts with newlines."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Line 1"},
                    {"kind": "text", "text": "Line 2"},
                    {"kind": "text", "text": "Line 3"}
                ]
            }
        }
        response = AgentResponse(raw)
        assert response.text == "Line 1\nLine 2\nLine 3"

    def test_text_property_empty_response(self) -> None:
        """Should return empty string for response without text parts."""
        raw = {"message": {"parts": []}}
        response = AgentResponse(raw)
        assert response.text == ""

    def test_text_property_caching(self) -> None:
        """Should cache text property for performance."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Cached"}
                ]
            }
        }
        response = AgentResponse(raw)

        # First access
        text1 = response.text
        # Second access (should use cache)
        text2 = response.text

        assert text1 == text2 == "Cached"
        assert response._text_cache == "Cached"

    def test_data_property_extracts_first_data_part(self) -> None:
        """Should extract data from first data part."""
        test_data = {"key": "value", "number": 42}
        raw = {
            "message": {
                "parts": [
                    {"kind": "data", "data": test_data}
                ]
            }
        }
        response = AgentResponse(raw)
        assert response.data == test_data

    def test_data_property_returns_none_when_no_data(self) -> None:
        """Should return None when no data parts exist."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Just text"}
                ]
            }
        }
        response = AgentResponse(raw)
        assert response.data is None

    def test_data_property_caching(self) -> None:
        """Should cache data property for performance."""
        test_data = {"cached": True}
        raw = {
            "message": {
                "parts": [
                    {"kind": "data", "data": test_data}
                ]
            }
        }
        response = AgentResponse(raw)

        # First access
        data1 = response.data
        # Second access (should use cache)
        data2 = response.data

        assert data1 == data2 == test_data
        assert response._data_cache == test_data

    def test_parts_property_returns_all_parts(self) -> None:
        """Should return all parts from response."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Hello"},
                    {"kind": "data", "data": {"x": 1}},
                    {"kind": "text", "text": "World"}
                ]
            }
        }
        response = AgentResponse(raw)
        parts = response.parts

        assert len(parts) == 3
        assert parts[0]["kind"] == "text"
        assert parts[1]["kind"] == "data"
        assert parts[2]["kind"] == "text"

    def test_parts_property_empty_response(self) -> None:
        """Should return empty list for response without parts."""
        raw = {"message": {"parts": []}}
        response = AgentResponse(raw)
        assert response.parts == []

    def test_str_method_returns_text(self) -> None:
        """String representation should return text property."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "String output"}
                ]
            }
        }
        response = AgentResponse(raw)
        assert str(response) == "String output"

    def test_repr_method_shows_text_and_data(self) -> None:
        """Repr should show text and data for debugging."""
        raw = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Text"},
                    {"kind": "data", "data": {"x": 1}}
                ]
            }
        }
        response = AgentResponse(raw)
        repr_str = repr(response)

        assert "AgentResponse" in repr_str
        assert "Text" in repr_str
        assert "{'x': 1}" in repr_str


class TestInterAgentClientURLResolution:
    """Tests for URL resolution logic in InterAgentClient."""

    def test_resolve_local_agent_path(self) -> None:
        """Should resolve /path to local base URL."""
        client = InterAgentClient(
            agent_identifier="/processor",
            local_base_url="http://localhost:8501"
        )
        assert client.target_url == "http://localhost:8501/processor"

    def test_resolve_local_agent_path_with_trailing_slash(self) -> None:
        """Should handle base URL with trailing slash."""
        client = InterAgentClient(
            agent_identifier="/analyzer",
            local_base_url="http://localhost:8501/"
        )
        assert client.target_url == "http://localhost:8501/analyzer"

    def test_resolve_direct_https_url(self) -> None:
        """Should pass through direct HTTPS URLs."""
        url = "https://api.example.com/agent"
        client = InterAgentClient(agent_identifier=url)
        assert client.target_url == url

    def test_resolve_direct_http_url(self) -> None:
        """Should pass through direct HTTP URLs."""
        url = "http://internal-service:8080/agent"
        client = InterAgentClient(agent_identifier=url)
        assert client.target_url == url

    def test_resolve_agent_name_from_registry(self) -> None:
        """Should resolve agent name from registry."""
        registry = {
            "data-processor": "https://processor.example.com",
            "analyzer": "https://analyzer.example.com"
        }
        client = InterAgentClient(
            agent_identifier="data-processor",
            agent_registry=registry
        )
        assert client.target_url == "https://processor.example.com"

    def test_resolve_unregistered_agent_name_raises_error(self) -> None:
        """Should raise ValueError for unregistered agent name."""
        registry = {"known-agent": "https://example.com"}

        with pytest.raises(ValueError) as exc_info:
            InterAgentClient(
                agent_identifier="unknown-agent",
                agent_registry=registry
            )

        assert "Could not resolve agent identifier 'unknown-agent'" in str(exc_info.value)
        assert "AGENT_REGISTRY" in str(exc_info.value)

    def test_load_agent_registry_from_env(self) -> None:
        """Should load agent registry from AGENT_REGISTRY environment variable."""
        registry_json = json.dumps({
            "agent1": "https://agent1.example.com",
            "agent2": "https://agent2.example.com"
        })

        with patch.dict(os.environ, {"AGENT_REGISTRY": registry_json}):
            registry = InterAgentClient._load_agent_registry()
            assert registry == {
                "agent1": "https://agent1.example.com",
                "agent2": "https://agent2.example.com"
            }

    def test_load_agent_registry_empty_env(self) -> None:
        """Should return empty dict when AGENT_REGISTRY not set."""
        with patch.dict(os.environ, {}, clear=True):
            registry = InterAgentClient._load_agent_registry()
            assert registry == {}

    def test_load_agent_registry_invalid_json(self) -> None:
        """Should handle invalid JSON gracefully."""
        with patch.dict(os.environ, {"AGENT_REGISTRY": "not-valid-json"}):
            registry = InterAgentClient._load_agent_registry()
            assert registry == {}

    def test_load_agent_registry_non_dict(self) -> None:
        """Should handle non-dict JSON gracefully."""
        with patch.dict(os.environ, {"AGENT_REGISTRY": '["not", "a", "dict"]'}):
            registry = InterAgentClient._load_agent_registry()
            assert registry == {}

    def test_default_local_base_url_from_env(self) -> None:
        """Should use LOCAL_AGENT_BASE_URL from environment."""
        with patch.dict(os.environ, {"LOCAL_AGENT_BASE_URL": "http://custom:9000"}):
            client = InterAgentClient(agent_identifier="/test")
            assert client.target_url == "http://custom:9000/test"

    def test_default_local_base_url_fallback(self) -> None:
        """Should fallback to localhost:8501 when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            client = InterAgentClient(agent_identifier="/test")
            assert client.target_url == "http://localhost:8501/test"


class TestInterAgentClientRetryLogic:
    """Tests for retry logic in InterAgentClient."""

    def test_should_retry_timeout_error(self) -> None:
        """Should retry timeout errors."""
        client = InterAgentClient(agent_identifier="/test")
        error = httpx.TimeoutException("Request timed out")
        assert client._should_retry(error) is True

    def test_should_retry_5xx_server_errors(self) -> None:
        """Should retry 5xx server errors."""
        client = InterAgentClient(agent_identifier="/test")

        # Create mock response with 500 error
        response = MagicMock()
        response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)

        assert client._should_retry(error) is True

    def test_should_retry_503_service_unavailable(self) -> None:
        """Should retry 503 service unavailable."""
        client = InterAgentClient(agent_identifier="/test")

        response = MagicMock()
        response.status_code = 503
        error = httpx.HTTPStatusError("Service unavailable", request=MagicMock(), response=response)

        assert client._should_retry(error) is True

    def test_should_not_retry_4xx_client_errors(self) -> None:
        """Should not retry 4xx client errors."""
        client = InterAgentClient(agent_identifier="/test")

        response = MagicMock()
        response.status_code = 400
        error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)

        assert client._should_retry(error) is False

    def test_should_not_retry_401_unauthorized(self) -> None:
        """Should not retry 401 unauthorized."""
        client = InterAgentClient(agent_identifier="/test")

        response = MagicMock()
        response.status_code = 401
        error = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=response)

        assert client._should_retry(error) is False

    def test_should_not_retry_404_not_found(self) -> None:
        """Should not retry 404 not found."""
        client = InterAgentClient(agent_identifier="/test")

        response = MagicMock()
        response.status_code = 404
        error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)

        assert client._should_retry(error) is False

    def test_should_retry_connect_error(self) -> None:
        """Should retry connection errors."""
        client = InterAgentClient(agent_identifier="/test")
        error = httpx.ConnectError("Connection failed")
        assert client._should_retry(error) is True

    def test_should_retry_network_error(self) -> None:
        """Should retry network errors."""
        client = InterAgentClient(agent_identifier="/test")
        error = httpx.NetworkError("Network unreachable")
        assert client._should_retry(error) is True

    def test_should_not_retry_other_errors(self) -> None:
        """Should not retry unknown errors."""
        client = InterAgentClient(agent_identifier="/test")
        error = ValueError("Some other error")
        assert client._should_retry(error) is False

    @pytest.mark.asyncio
    async def test_call_with_retry_success_first_attempt(self) -> None:
        """Should succeed on first attempt without retry."""
        client = InterAgentClient(agent_identifier="/test", timeout=10.0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await client._call_with_retry(
                "POST",
                "http://example.com/test",
                json={"test": "data"}
            )

            assert response.status_code == 200
            assert mock_client.request.call_count == 1

    @pytest.mark.asyncio
    async def test_call_with_retry_succeeds_after_transient_error(self) -> None:
        """Should retry and succeed after transient error."""
        client = InterAgentClient(
            agent_identifier="/test",
            timeout=10.0,
            max_retries=2,
            retry_backoff_base=0.01  # Fast retry for tests
        )

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client

            # First attempt: timeout, second attempt: success
            mock_client.request.side_effect = [
                httpx.TimeoutException("Timeout"),
                mock_success_response
            ]

            mock_client_class.return_value = mock_client

            response = await client._call_with_retry(
                "POST",
                "http://example.com/test",
                json={"test": "data"}
            )

            assert response.status_code == 200
            assert mock_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_call_with_retry_fails_after_max_retries(self) -> None:
        """Should raise error after exhausting max retries."""
        client = InterAgentClient(
            agent_identifier="/test",
            timeout=10.0,
            max_retries=2,
            retry_backoff_base=0.01  # Fast retry for tests
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client

            # All attempts fail with timeout
            mock_client.request.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.TimeoutException):
                await client._call_with_retry(
                    "POST",
                    "http://example.com/test",
                    json={"test": "data"}
                )

            # Should try initial + 2 retries = 3 total
            assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_call_with_retry_no_retry_for_4xx(self) -> None:
        """Should not retry 4xx errors."""
        client = InterAgentClient(
            agent_identifier="/test",
            timeout=10.0,
            max_retries=2
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client

            response = MagicMock()
            response.status_code = 400
            error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
            mock_client.request.side_effect = error
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await client._call_with_retry(
                    "POST",
                    "http://example.com/test",
                    json={"test": "data"}
                )

            # Should only try once (no retries for 4xx)
            assert mock_client.request.call_count == 1


class TestInterAgentClientCalls:
    """Tests for call() and call_with_data() methods."""

    @pytest.mark.asyncio
    async def test_call_sends_text_message(self) -> None:
        """Should send text message with correct A2A format."""
        client = InterAgentClient(
            agent_identifier="https://example.com/agent",
            timeout=10.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "parts": [{"kind": "text", "text": "Response text"}]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await client.call("Test message")

            assert response.text == "Response text"
            assert mock_client.request.call_count == 1

            # Verify request format
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/agent/a2a/v1/messages"

            request_json = call_args[1]["json"]
            assert request_json["role"] == "user"
            assert len(request_json["parts"]) == 1
            assert request_json["parts"][0]["kind"] == "text"
            assert request_json["parts"][0]["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_call_with_custom_timeout(self) -> None:
        """Should use custom timeout when provided."""
        client = InterAgentClient(
            agent_identifier="https://example.com/agent",
            timeout=30.0  # Default timeout
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "parts": [{"kind": "text", "text": "Response"}]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            await client.call("Test", timeout=60.0)

            # Verify timeout was passed to AsyncClient constructor
            call_args = mock_client_class.call_args
            assert call_args[1]["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_call_with_data_sends_structured_data(self) -> None:
        """Should send structured data with DataPart."""
        client = InterAgentClient(
            agent_identifier="https://example.com/agent",
            timeout=10.0
        )

        test_data = {"query": "test", "limit": 10, "filters": ["a", "b"]}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "parts": [{"kind": "data", "data": {"result": "success"}}]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await client.call_with_data(test_data)

            assert response.data == {"result": "success"}

            # Verify request format
            call_args = mock_client.request.call_args
            request_json = call_args[1]["json"]
            assert request_json["role"] == "user"
            assert len(request_json["parts"]) == 1
            assert request_json["parts"][0]["kind"] == "data"
            assert request_json["parts"][0]["data"] == test_data

    @pytest.mark.asyncio
    async def test_call_with_auth_token_propagation(self) -> None:
        """Should include Authorization header when auth_token provided."""
        client = InterAgentClient(
            agent_identifier="https://example.com/agent",
            auth_token="jwt-token-abc123",
            timeout=10.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "parts": [{"kind": "text", "text": "Authenticated response"}]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            await client.call("Test message")

            # Verify Authorization header
            call_args = mock_client.request.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer jwt-token-abc123"

    @pytest.mark.asyncio
    async def test_call_without_auth_token(self) -> None:
        """Should not include Authorization header when no auth_token."""
        client = InterAgentClient(
            agent_identifier="https://example.com/agent",
            auth_token=None,
            timeout=10.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "parts": [{"kind": "text", "text": "Public response"}]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            await client.call("Test message")

            # Verify no Authorization header
            call_args = mock_client.request.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_close_method(self) -> None:
        """Close method should complete without error."""
        client = InterAgentClient(agent_identifier="/test")
        await client.close()  # Should not raise


class TestInterAgentClientIntegration:
    """Integration tests for InterAgentClient."""

    @pytest.mark.asyncio
    async def test_end_to_end_local_agent_call(self) -> None:
        """Should successfully call local agent end-to-end."""
        with patch.dict(os.environ, {"LOCAL_AGENT_BASE_URL": "http://localhost:8501"}):
            client = InterAgentClient(
                agent_identifier="/calculator",
                auth_token="test-jwt-token",
                timeout=10.0,
                max_retries=1,
                retry_backoff_base=0.01
            )

            assert client.target_url == "http://localhost:8501/calculator"

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "parts": [
                        {"kind": "text", "text": "Result: 42"}
                    ]
                }
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.request.return_value = mock_response
                mock_client_class.return_value = mock_client

                response = await client.call("Calculate 40 + 2")

                assert response.text == "Result: 42"

                # Verify full request
                call_args = mock_client.request.call_args
                assert call_args[0][1] == "http://localhost:8501/calculator/a2a/v1/messages"
                assert call_args[1]["headers"]["Authorization"] == "Bearer test-jwt-token"

    @pytest.mark.asyncio
    async def test_end_to_end_registry_agent_call(self) -> None:
        """Should successfully call registry agent end-to-end."""
        registry_json = json.dumps({
            "data-processor": "https://processor.healthuniverse.com",
            "analyzer": "https://analyzer.healthuniverse.com"
        })

        with patch.dict(os.environ, {"AGENT_REGISTRY": registry_json}):
            client = InterAgentClient(
                agent_identifier="data-processor",
                timeout=10.0
            )

            assert client.target_url == "https://processor.healthuniverse.com"

            test_data = {"operation": "analyze", "data": [1, 2, 3]}

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "parts": [
                        {"kind": "data", "data": {"mean": 2.0, "sum": 6}}
                    ]
                }
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.request.return_value = mock_response
                mock_client_class.return_value = mock_client

                response = await client.call_with_data(test_data)

                assert response.data == {"mean": 2.0, "sum": 6}

                # Verify request
                call_args = mock_client.request.call_args
                assert call_args[0][1] == "https://processor.healthuniverse.com/a2a/v1/messages"
                request_json = call_args[1]["json"]
                assert request_json["parts"][0]["data"] == test_data
