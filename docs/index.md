# Health Universe A2A SDK

A simple, batteries-included SDK for building A2A-compliant agents on the Health Universe platform.

## Features

- **Background Jobs**: Long-running tasks with progress updates via `Agent`
- **Document Operations**: Read and write documents via `context.document_client`
- **Inter-Agent Communication**: Call other agents with `InterAgentClient`
- **Extension Support**: FILE_ACCESS v2, BACKGROUND_JOB, LOG_LEVEL extensions

## Quick Example

```python
from health_universe_a2a import Agent, AgentContext

class MyAgent(Agent):
    def get_agent_name(self) -> str:
        return "my-agent"

    def get_agent_description(self) -> str:
        return "A simple agent"

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Processing your request...", progress=0.5)
        return f"Result: {message}"

# Run the agent
if __name__ == "__main__":
    MyAgent().serve()
```

## Installation

```bash
pip install health-universe-a2a
```

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions.

## Architecture

The SDK provides two main classes:

| Class | Use Case | Description |
|-------|----------|-------------|
| `Agent` | All agents | Main class for building agents (alias for `AsyncAgent`) |
| `A2AAgentBase` | Custom implementations | Abstract base class for advanced customization |

## Next Steps

- [Quick Start](getting-started/quickstart.md) - Build your first agent
- [Background Jobs](guides/background-jobs.md) - Long-running tasks with progress updates
- [API Reference](api/agents.md) - Full API documentation
