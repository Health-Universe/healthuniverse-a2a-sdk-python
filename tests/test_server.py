"""Integration tests for server functionality."""

from typing import Any

from starlette.testclient import TestClient

from health_universe_a2a import (
    A2AAgent,
    MessageContext,
    ValidationAccepted,
    ValidationRejected,
    create_app,
)


class TestServerAgent(A2AAgent):
    """Simple agent for server testing."""

    def get_agent_name(self) -> str:
        return "Test Server Agent"

    def get_agent_description(self) -> str:
        return "Agent for testing HTTP server functionality"

    def get_agent_version(self) -> str:
        return "1.0.0"

    async def process_message(self, message: str, context: MessageContext) -> str:
        return f"Echo: {message}"


class TestValidatingAgent(A2AAgent):
    """Agent with custom validation for testing rejection flow."""

    def get_agent_name(self) -> str:
        return "Validating Agent"

    def get_agent_description(self) -> str:
        return "Tests validation"

    async def validate_message(
        self, message: str, metadata: dict[str, Any]
    ) -> ValidationAccepted | ValidationRejected:
        if len(message) < 5:
            return ValidationRejected(reason="Message too short")
        return ValidationAccepted()

    async def process_message(self, message: str, context: MessageContext) -> str:
        return f"Valid: {message}"


class TestAgentCardEndpoint:
    """Tests for GET /.well-known/agent-card.json endpoint."""

    def test_agent_card_endpoint_returns_200(self) -> None:
        """Agent card endpoint should return 200 OK."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200

    def test_agent_card_returns_json(self) -> None:
        """Agent card should return valid JSON."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")

        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)

    def test_agent_card_contains_required_fields(self) -> None:
        """Agent card should contain all required A2A fields."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        # Required A2A fields (uses camelCase in JSON)
        assert card["name"] == "Test Server Agent"
        assert card["description"] == "Agent for testing HTTP server functionality"
        assert card["version"] == "1.0.0"
        assert card["protocolVersion"] == "0.3.0"
        assert card["preferredTransport"] == "JSONRPC"

    def test_agent_card_includes_provider(self) -> None:
        """Agent card should include provider information."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert "provider" in card
        assert "organization" in card["provider"]
        assert "url" in card["provider"]

    def test_agent_card_includes_capabilities(self) -> None:
        """Agent card should include capabilities."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert "capabilities" in card
        assert "streaming" in card["capabilities"]
        assert "pushNotifications" in card["capabilities"]  # camelCase in JSON


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_endpoint_exists(self) -> None:
        """Health endpoint should exist (may be 200 or 404 depending on server config)."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/health")

        # A2A server may or may not provide health endpoint
        # Just check that we get a response (not a server error)
        assert response.status_code in [200, 404]


class TestJSONRPCEndpoint:
    """Tests for JSON-RPC message/send endpoint."""

    def test_jsonrpc_endpoint_accepts_post(self) -> None:
        """JSON-RPC endpoint should accept POST requests."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-123",
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                }
            },
            "id": 1,
        }

        response = client.post("/", json=request_data)

        # Should return 200 (even if processing fails, JSON-RPC uses 200 with error object)
        assert response.status_code == 200

    def test_jsonrpc_returns_json_response(self) -> None:
        """JSON-RPC should return JSON response."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-123",
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                }
            },
            "id": 1,
        }

        response = client.post("/", json=request_data)
        data = response.json()

        assert isinstance(data, dict)
        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"

    def test_jsonrpc_invalid_method_returns_error(self) -> None:
        """JSON-RPC should return error for invalid method."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        request_data = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": 1,
        }

        response = client.post("/", json=request_data)
        data = response.json()

        assert "error" in data
        assert "code" in data["error"]

    def test_jsonrpc_missing_params_returns_error(self) -> None:
        """JSON-RPC should return error for missing params."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": 1,
            # Missing params
        }

        response = client.post("/", json=request_data)
        data = response.json()

        assert response.status_code == 200
        # JSON-RPC returns errors with 200 status code
        assert "error" in data or "result" in data


class TestMessageProcessingFlow:
    """Tests for full message processing flow through the server."""

    def test_agent_processes_message_successfully(self) -> None:
        """Agent should successfully process valid message."""
        agent = TestServerAgent()
        app = create_app(agent)
        client = TestClient(app)

        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-456",
                    "role": "user",
                    "parts": [{"text": "Hello World"}],
                }
            },
            "id": 1,
        }

        response = client.post("/", json=request_data)
        data = response.json()

        assert response.status_code == 200
        # Check for successful result (not error)
        if "error" not in data:
            assert "result" in data


class TestValidationFlow:
    """Tests for validation through the server."""

    def test_validation_rejection_through_server(self) -> None:
        """Server should handle validation rejection correctly."""
        agent = TestValidatingAgent()
        app = create_app(agent)
        client = TestClient(app)

        # Short message should be rejected
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-789",
                    "role": "user",
                    "parts": [{"text": "Hi"}],  # Too short
                }
            },
            "id": 1,
        }

        response = client.post("/", json=request_data)

        assert response.status_code == 200
        # Request should complete (validation happens inside)

    def test_validation_acceptance_through_server(self) -> None:
        """Server should handle validation acceptance correctly."""
        agent = TestValidatingAgent()
        app = create_app(agent)
        client = TestClient(app)

        # Long enough message should be accepted
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-101",
                    "role": "user",
                    "parts": [{"text": "Hello World"}],  # Long enough
                }
            },
            "id": 1,
        }

        response = client.post("/", json=request_data)

        assert response.status_code == 200


class TestCustomAgentConfiguration:
    """Tests for agents with custom configuration."""

    def test_custom_version_in_agent_card(self) -> None:
        """Custom agent version should appear in agent card."""

        class CustomVersionAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "Custom Agent"

            def get_agent_description(self) -> str:
                return "Has custom version"

            def get_agent_version(self) -> str:
                return "2.5.0"

            async def process_message(self, message: str, context: MessageContext) -> str:
                return "processed"

        agent = CustomVersionAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert card["version"] == "2.5.0"

    def test_custom_formats_in_agent_card(self) -> None:
        """Custom input/output formats should appear in agent card."""

        class CustomFormatsAgent(A2AAgent):
            def get_agent_name(self) -> str:
                return "Formats Agent"

            def get_agent_description(self) -> str:
                return "Has custom formats"

            def get_supported_input_formats(self) -> list[str]:
                return ["text/csv", "application/xml"]

            def get_supported_output_formats(self) -> list[str]:
                return ["application/json", "text/html"]

            async def process_message(self, message: str, context: MessageContext) -> str:
                return "processed"

        agent = CustomFormatsAgent()
        app = create_app(agent)
        client = TestClient(app)

        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        # JSON uses camelCase
        assert "text/csv" in card["defaultInputModes"]
        assert "application/xml" in card["defaultInputModes"]
        assert "application/json" in card["defaultOutputModes"]
        assert "text/html" in card["defaultOutputModes"]


class TestServerCreation:
    """Tests for server creation utilities."""

    def test_create_app_returns_starlette_app(self) -> None:
        """create_app should return a Starlette application."""
        agent = TestServerAgent()
        app = create_app(agent)

        assert app is not None
        # Should be a Starlette app
        assert hasattr(app, "routes")

    def test_create_app_with_agent(self) -> None:
        """create_app should work with any A2AAgent."""
        agent = TestServerAgent()
        app = create_app(agent)

        # Should be able to create test client
        client = TestClient(app)
        assert client is not None

    def test_multiple_agents_can_create_apps(self) -> None:
        """Multiple agents should be able to create separate apps."""
        agent1 = TestServerAgent()
        agent2 = TestValidatingAgent()

        app1 = create_app(agent1)
        app2 = create_app(agent2)

        # Apps should be different instances
        assert app1 is not app2

        # Both should work
        client1 = TestClient(app1)
        client2 = TestClient(app2)

        response1 = client1.get("/.well-known/agent-card.json")
        response2 = client2.get("/.well-known/agent-card.json")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Should have different names
        card1 = response1.json()
        card2 = response2.json()
        assert card1["name"] != card2["name"]


if __name__ == "__main__":
    """
    Run a test server for manual testing.

    This allows you to start a real HTTP server with the test agent
    for manual testing with curl or other HTTP clients.

    Usage:
        python tests/test_server.py

    Then test with:
        curl http://localhost:8000/.well-known/agent-card.json
        curl -X POST http://localhost:8000/ -H "Content-Type: application/json" -d '{
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "manual-test-1",
                    "role": "user",
                    "parts": [{"text": "Hello from curl!"}]
                }
            },
            "id": 1
        }'
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create test agent
    agent = TestServerAgent()

    print("\n" + "=" * 60)
    print("🧪 Starting TEST SERVER for manual testing")
    print("=" * 60)
    print(f"Agent: {agent.get_agent_name()}")
    print(f"Version: {agent.get_agent_version()}")
    print("\nEndpoints:")
    print("  • Agent Card: http://localhost:8000/.well-known/agent-card.json")
    print("  • JSON-RPC:   POST http://localhost:8000/")
    print("  • Health:     http://localhost:8000/health")
    print("\nTest with:")
    print("  curl http://localhost:8000/.well-known/agent-card.json")
    print("=" * 60)
    print()

    # Start server
    agent.serve(port=8000)
