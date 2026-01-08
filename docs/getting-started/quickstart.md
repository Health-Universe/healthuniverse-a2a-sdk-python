# Quick Start

This guide walks you through creating your first A2A agent.

## Creating Your First Agent

```python
from health_universe_a2a import Agent, AgentContext

class EchoAgent(Agent):
    def get_agent_name(self) -> str:
        return "echo-agent"

    def get_agent_description(self) -> str:
        return "Echoes back the user's message"

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Processing your message...")
        return f"You said: {message}"

# Run the agent server
if __name__ == "__main__":
    EchoAgent().serve()
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

## Adding Progress Updates

Keep users informed during long-running tasks:

```python
from health_universe_a2a import Agent, AgentContext, UpdateImportance

class AnalysisAgent(Agent):
    def get_agent_name(self) -> str:
        return "analysis-agent"

    def get_agent_description(self) -> str:
        return "Analyzes data with progress updates"

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Starting analysis...", progress=0.0)

        # Do some work
        await context.update_progress("Processing data...", progress=0.5)

        # Important milestone - pushed to Navigator UI
        await context.update_progress(
            "Analysis complete!",
            progress=1.0,
            importance=UpdateImportance.NOTICE
        )

        return "Analysis finished successfully"
```

## Next Steps

- [Background Jobs](../guides/background-jobs.md) - Handle long-running tasks
- [Inter-Agent Communication](../guides/inter-agent.md) - Call other agents
