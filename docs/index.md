# Health Universe A2A SDK

A simple, batteries-included SDK for building A2A-compliant agents on the Health Universe platform.

## Features

- **Streaming Agents**: Real-time streaming responses with `StreamingAgent`
- **Background Jobs**: Long-running tasks with progress updates via `AsyncAgent`
- **Storage Integration**: S3 storage backend with `directory_context` for file handling
- **Inter-Agent Communication**: Call other agents with `InterAgentClient`
- **Extension Support**: FILE_ACCESS v2, BACKGROUND_JOB, LOG_LEVEL extensions

## Quick Example

```python
from health_universe_a2a import StreamingAgent, StreamingContext

class MyAgent(StreamingAgent):
    name = "my-agent"
    description = "A simple streaming agent"

    async def stream(self, query: str, context: StreamingContext):
        yield "Processing your request..."
        # Your agent logic here
        yield f"Result: {query}"

# Run the agent
if __name__ == "__main__":
    from health_universe_a2a import serve
    serve(MyAgent())
```

## Installation

```bash
pip install health-universe-a2a
```

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions.

## Architecture

The SDK provides three main agent types:

| Agent Type | Use Case | Response Style |
|------------|----------|----------------|
| `StreamingAgent` | Real-time interactions | SSE streaming |
| `AsyncAgent` | Long-running tasks | Background webhook updates |
| `A2AAgentBase` | Custom implementations | Flexible |

## Next Steps

- [Quick Start](getting-started/quickstart.md) - Build your first agent
- [Streaming Agents](guides/streaming-agents.md) - Real-time responses
- [Background Jobs](guides/background-jobs.md) - Long-running tasks
- [API Reference](api/agents.md) - Full API documentation
