# Quick Start

This guide walks you through creating your first A2A agent.

## Creating a Streaming Agent

The simplest way to get started is with a `StreamingAgent`:

```python
from health_universe_a2a import StreamingAgent, StreamingContext

class EchoAgent(StreamingAgent):
    name = "echo-agent"
    description = "Echoes back the user's message"

    async def stream(self, query: str, context: StreamingContext):
        yield f"You said: {query}"

# Run the agent server
if __name__ == "__main__":
    from health_universe_a2a import serve
    serve(EchoAgent())
```

Run it:

```bash
python my_agent.py
```

Your agent is now running at `http://localhost:8000`.

## Testing Your Agent

### View the Agent Card

```bash
curl http://localhost:8000/.well-known/agent.json
```

### Send a Message

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "Hello!"}]}}}'
```

## Adding Skills

Define what your agent can do:

```python
from health_universe_a2a import StreamingAgent, StreamingContext, AgentSkill

class CalculatorAgent(StreamingAgent):
    name = "calculator"
    description = "Performs basic math operations"
    skills = [
        AgentSkill(
            id="add",
            name="Addition",
            description="Add two numbers together"
        ),
        AgentSkill(
            id="multiply",
            name="Multiplication",
            description="Multiply two numbers"
        )
    ]

    async def stream(self, query: str, context: StreamingContext):
        # Parse and handle the query
        yield f"Calculating: {query}"
```

## Next Steps

- [Streaming Agents](../guides/streaming-agents.md) - Learn about streaming responses
- [Background Jobs](../guides/background-jobs.md) - Handle long-running tasks
- [Storage](../guides/storage.md) - Work with files and S3 storage
