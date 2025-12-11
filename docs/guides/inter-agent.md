# Inter-Agent Communication

Agents can call other agents using the `InterAgentClient` or the built-in `call_agent()` method.

## Using call_agent()

The simplest way to call another agent:

```python
from health_universe_a2a import StreamingAgent, StreamingContext

class OrchestratorAgent(StreamingAgent):
    async def stream(self, query: str, context: StreamingContext):
        # Call by URL
        result = await self.call_agent(
            "https://other-agent.healthuniverse.com",
            query
        )
        yield f"Other agent said: {result}"

        # Call by registry name
        result = await self.call_agent("summarizer", query)
        yield f"Summary: {result}"
```

## Agent Registry

Register agents by name for easier discovery:

```python
from health_universe_a2a import AgentRegistry, get_agent_registry

# Load from file or environment
registry = get_agent_registry()

# Or configure programmatically
registry = AgentRegistry()
registry._registry = {
    "summarizer": "https://summarizer.healthuniverse.com",
    "analyzer": "https://analyzer.healthuniverse.com"
}
```

Registry configuration via `AGENT_REGISTRY` environment variable:

```bash
export AGENT_REGISTRY='{"summarizer": "http://localhost:8001", "analyzer": "http://localhost:8002"}'
```

Or via file at `~/.a2a/registry.json`:

```json
{
  "summarizer": "https://summarizer.healthuniverse.com",
  "analyzer": "https://analyzer.healthuniverse.com"
}
```

## InterAgentClient

For more control over inter-agent communication:

```python
from health_universe_a2a import InterAgentClient

# Create from registry
client = InterAgentClient.from_registry("summarizer")

# Or with explicit URL
client = InterAgentClient(base_url="https://agent.example.com")

# Send a message
response = await client.send_message({
    "role": "user",
    "parts": [{"kind": "text", "text": "Summarize this..."}]
})
```

## JWT Propagation

The SDK automatically propagates authentication when calling other agents:

```python
# JWT from incoming request is forwarded
result = await self.call_agent("other-agent", query)
# Authorization header is automatically included
```

## Timeout Handling

```python
result = await self.call_agent(
    "slow-agent",
    query,
    timeout=60.0  # 60 second timeout
)
```

## Response Handling

The `call_agent()` method handles various response formats:

```python
# Returns text content for simple responses
result = await self.call_agent("agent", "query")
# result is a string

# For complex responses, returns the parsed message
result = await self.call_agent("agent", {"complex": "input"})
# result may be a dict or Message object
```
