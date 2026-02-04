# Inter-Agent Communication

Agents can call other A2A-compliant agents using the built-in `call_agent()` method.

## Using call_agent()

The simplest way to call another agent:

```python
from health_universe_a2a import Agent, AgentContext

class OrchestratorAgent(Agent):
    def get_agent_name(self) -> str:
        return "orchestrator"

    def get_agent_description(self) -> str:
        return "Orchestrates multiple agents"

    async def process_message(self, message: str, context: AgentContext) -> str:
        # Call by URL
        result = await self.call_agent(
            "https://other-agent.healthuniverse.com",
            message,
            context
        )

        # Call by local path (uses LOCAL_AGENT_BASE_URL)
        result = await self.call_agent("/summarizer", message, context)

        # Call by registry name
        result = await self.call_agent("analyzer", message, context)

        return f"Result: {result}"
```

## Agent Identifier Formats

1. **Local agent path**: `/agent-name` - Uses `LOCAL_AGENT_BASE_URL` (default: `http://localhost:8501`)
2. **Direct URL**: `https://...` - Calls directly with HTTPS
3. **Registry name**: `agent-name` - Looks up in `AGENT_REGISTRY` environment variable

## Calling with Structured Data

Send dictionaries or lists instead of plain text:

```python
async def process_message(self, message: str, context: AgentContext) -> str:
    # Call with structured data
    result = await self.call_agent(
        "/processor",
        {"document_path": "/path/to/doc.pdf", "options": {"format": "json"}},
        context
    )

    return f"Processed: {result}"
```

## Agent Registry

Configure agent URLs via the `AGENT_REGISTRY` environment variable:

```bash
export AGENT_REGISTRY='{"summarizer": "http://localhost:8001", "analyzer": "http://localhost:8002"}'
```

Then call agents by name:

```python
result = await self.call_agent("summarizer", message, context)
```

## JWT Propagation

The SDK automatically propagates authentication when calling other agents:

```python
# JWT from incoming request is forwarded automatically
result = await self.call_agent("other-agent", message, context)
# Authorization header is automatically included
```

## Timeout Handling

```python
result = await self.call_agent(
    "slow-agent",
    message,
    context,
    timeout=60.0  # 60 second timeout
)
```

## Response Handling

The `call_agent()` method parses responses automatically:

```python
# Returns text content for simple responses
result = await self.call_agent("agent", "query", context)
# result is a string

# For complex responses with artifacts
result = await self.call_agent("agent", {"complex": "input"}, context)
# result may be text, dict, or structured data
```

## Lower-Level Access

For more control, use `call_other_agent()` which returns the full `AgentResponse`:

```python
response = await self.call_other_agent("/processor", message, context)

# Access response properties
text = response.text           # Text content
data = response.data           # Parsed JSON data
parts = response.parts         # Message parts
raw = response.raw_response    # Full raw response
```

Or for structured data:

```python
response = await self.call_other_agent_with_data(
    "/data-processor",
    {"query": message, "format": "json"},
    context
)
```
