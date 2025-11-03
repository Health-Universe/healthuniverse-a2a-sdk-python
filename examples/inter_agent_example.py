"""
Inter-Agent Communication Example

Demonstrates:
- Calling local agents (same pod, bypasses ingress)
- Calling remote agents with JWT propagation
- Structured data exchange between agents
- Handling agent responses (text, data, raw)
- Error handling and retries
- Multiple agent orchestration
"""

import asyncio
import json
from typing import Any

from health_universe_a2a import (
    StreamingAgent,
    MessageContext,
    AgentResponse,
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)


class OrchestratorAgent(StreamingAgent):
    """
    Orchestrator agent that coordinates multiple agents.

    This agent demonstrates inter-agent communication patterns:
    - Calling local agents (same pod)
    - Calling remote agents
    - Sequential and parallel agent calls
    - Error handling and fallbacks
    """

    def get_agent_name(self) -> str:
        return "Orchestrator Agent"

    def get_agent_description(self) -> str:
        return "Coordinates multiple agents to process complex requests"

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_provider_organization(self) -> str:
        return "Example Multi-Agent Systems"

    def get_provider_url(self) -> str:
        return "https://example.com"

    async def validate_message(
        self, message: str, metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate that message has a valid command."""
        try:
            request = json.loads(message)
            command = request.get("command")

            if not command:
                return ValidationRejected(reason="Missing 'command' field in request")

            valid_commands = ["analyze", "process", "transform"]
            if command not in valid_commands:
                return ValidationRejected(
                    reason=f"Invalid command '{command}'. Must be one of: {valid_commands}"
                )

            return ValidationAccepted(estimated_duration_seconds=60)

        except json.JSONDecodeError:
            return ValidationRejected(reason="Request must be valid JSON")

    async def process_message(self, message: str, context: MessageContext) -> str:
        """
        Process request by coordinating multiple agents.

        Expected message format:
        {
            "command": "analyze" | "process" | "transform",
            "data": "input data",
            "options": {...}
        }
        """
        # Parse request
        await context.update_progress("Parsing request...", 0.1)
        request = json.loads(message)
        command = request["command"]
        data = request.get("data", "")
        options = request.get("options", {})

        # Route to appropriate workflow
        if command == "analyze":
            return await self._analyze_workflow(data, options, context)
        elif command == "process":
            return await self._process_workflow(data, options, context)
        elif command == "transform":
            return await self._transform_workflow(data, options, context)

        return "Unknown command"

    async def _analyze_workflow(
        self, data: str, options: dict, context: MessageContext
    ) -> str:
        """
        Analyze workflow: preprocessor → analyzer → formatter

        Demonstrates sequential agent calls with local agents.
        """
        await context.update_progress("Starting analysis workflow...", 0.2)

        try:
            # Step 1: Preprocess with local agent
            await context.update_progress("Preprocessing data...", 0.3)
            self.logger.info("Calling local preprocessor agent")

            # Call local agent (same pod, bypasses ingress/egress)
            preprocessor_response = await self.call_other_agent(
                "/preprocessor",  # Local agent path
                data,
                context,
                timeout=30.0,
            )

            preprocessed_data = preprocessor_response.text
            self.logger.info(f"Preprocessed data length: {len(preprocessed_data)}")

            # Step 2: Analyze with structured data
            await context.update_progress("Analyzing...", 0.6)
            self.logger.info("Calling local analyzer agent")

            analyzer_response = await self.call_other_agent_with_data(
                "/analyzer",  # Local agent path
                {
                    "data": preprocessed_data,
                    "mode": options.get("analysis_mode", "standard"),
                },
                context,
                timeout=60.0,
            )

            # Extract structured results
            if analyzer_response.data:
                analysis_results = analyzer_response.data
                self.logger.info(f"Analysis complete: {analysis_results}")
            else:
                analysis_results = {"text": analyzer_response.text}

            # Step 3: Format results
            await context.update_progress("Formatting results...", 0.9)

            formatter_response = await self.call_other_agent_with_data(
                "/formatter",
                {"results": analysis_results, "format": options.get("output_format", "markdown")},
                context,
            )

            # Add analysis as artifact
            await context.add_artifact(
                name="Analysis Results",
                content=json.dumps(analysis_results, indent=2),
                data_type="application/json",
            )

            return formatter_response.text

        except Exception as e:
            self.logger.error(f"Analysis workflow failed: {e}")
            return f"Analysis failed: {str(e)}"

    async def _process_workflow(
        self, data: str, options: dict, context: MessageContext
    ) -> str:
        """
        Process workflow: parallel processors → merger

        Demonstrates parallel agent calls with error handling.
        """
        await context.update_progress("Starting parallel processing...", 0.2)

        # Call multiple processors in parallel
        processors = ["/processor-a", "/processor-b", "/processor-c"]
        results = []

        # Launch parallel calls
        await context.update_progress(f"Processing with {len(processors)} agents...", 0.3)

        tasks = []
        for processor in processors:
            task = self.call_other_agent(
                processor,
                data,
                context,
                timeout=45.0,
            )
            tasks.append(task)

        # Wait for all with error handling
        completed_results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                completed_results.append(result.text)
                await context.update_progress(
                    f"Completed {i+1}/{len(processors)} processors",
                    0.3 + (0.4 * (i + 1) / len(processors)),
                )
            except Exception as e:
                self.logger.warning(f"Processor {i} failed: {e}")
                completed_results.append(f"[FAILED: {e}]")

        # Merge results with local agent
        await context.update_progress("Merging results...", 0.8)

        merger_response = await self.call_other_agent_with_data(
            "/merger",
            {"results": completed_results},
            context,
        )

        return merger_response.text

    async def _transform_workflow(
        self, data: str, options: dict, context: MessageContext
    ) -> str:
        """
        Transform workflow: local + remote agents

        Demonstrates mixing local and remote agent calls.
        """
        await context.update_progress("Starting transformation...", 0.2)

        try:
            # Step 1: Transform locally
            await context.update_progress("Local transformation...", 0.3)

            local_response = await self.call_other_agent(
                "/transformer",  # Local agent
                data,
                context,
                timeout=30.0,
            )

            # Step 2: Validate with remote agent (if configured)
            remote_validator_url = options.get("remote_validator_url")
            if remote_validator_url:
                await context.update_progress("Remote validation...", 0.7)
                self.logger.info(f"Calling remote validator: {remote_validator_url}")

                try:
                    # Call remote agent with longer timeout
                    validator_response = await self.call_other_agent(
                        remote_validator_url,  # Remote agent URL
                        local_response.text,
                        context,
                        timeout=120.0,  # Remote agents may be slower
                    )

                    # Check validation result
                    if validator_response.data:
                        validation = validator_response.data
                        if not validation.get("valid", True):
                            return f"Transformation failed validation: {validation.get('reason')}"

                except Exception as e:
                    self.logger.warning(f"Remote validation failed: {e}")
                    # Continue without validation if remote agent unavailable
                    pass

            return f"Transformation complete: {local_response.text}"

        except Exception as e:
            self.logger.error(f"Transform workflow failed: {e}")
            return f"Transformation failed: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    print("Inter-Agent Communication Example")
    print("\nFeatures demonstrated:")
    print("  - Local agent calls (same pod, bypasses ingress)")
    print("  - Remote agent calls with JWT propagation")
    print("  - Sequential agent orchestration")
    print("  - Parallel agent calls")
    print("  - Structured data exchange")
    print("  - Error handling and fallbacks")
    print("  - AgentResponse access patterns")

    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    print("\nLocal agents (same pod at localhost:8501):")
    print("  /preprocessor   - Preprocesses input data")
    print("  /analyzer       - Analyzes preprocessed data")
    print("  /formatter      - Formats analysis results")
    print("  /processor-a/b/c - Parallel processors")
    print("  /merger         - Merges parallel results")
    print("  /transformer    - Transforms data")

    print("\nRemote agents:")
    print("  https://api.example.com/validator - Remote validation service")

    print("\nEnvironment variables:")
    print("  LOCAL_AGENT_BASE_URL - Base URL for local agents (default: http://localhost:8501)")
    print("  AGENT_REGISTRY - JSON map of agent names to URLs")
    print("    Example: AGENT_REGISTRY='{\"analyzer\": \"http://...\"}'")

    print("\n" + "=" * 70)
    print("Example requests:")
    print("=" * 70)

    print("\n1. Analysis workflow (sequential):")
    print(json.dumps({
        "command": "analyze",
        "data": "Sample data to analyze",
        "options": {
            "analysis_mode": "detailed",
            "output_format": "markdown"
        }
    }, indent=2))

    print("\n2. Process workflow (parallel):")
    print(json.dumps({
        "command": "process",
        "data": "Data for parallel processing",
        "options": {}
    }, indent=2))

    print("\n3. Transform workflow (local + remote):")
    print(json.dumps({
        "command": "transform",
        "data": "Data to transform",
        "options": {
            "remote_validator_url": "https://api.example.com/validator"
        }
    }, indent=2))

    print("\n" + "=" * 70)
    print("Key concepts:")
    print("=" * 70)
    print("\n- Local agents start with '/' and use localhost:8501")
    print("- Remote agents use full https:// URLs")
    print("- JWT automatically propagated from context.auth_token")
    print("- Auto-retry with exponential backoff for transient errors")
    print("- AgentResponse provides .text, .data, .parts, .raw_response")
    print("- Timeout configurable per call (default 30s)")
    print("- Max 3 retries with 1s, 2s, 4s backoff")
