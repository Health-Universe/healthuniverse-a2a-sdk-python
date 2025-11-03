# Health Universe A2A SDK Examples

This directory contains example agents demonstrating various features of the SDK.

## Installation

First, install the SDK with server support:

```bash
# From the project root
uv pip install -e ".[server]"
```

## Examples

### 1. Simple Echo Agent (`simple_agent.py`)

A minimal agent that demonstrates the basics:
- Creating an agent by subclassing `A2AAgent`
- Implementing required methods
- Starting an HTTP server with `agent.serve()`

**Run it:**
```bash
python examples/simple_agent.py
```

**Test it:**
```bash
# View agent card
curl http://localhost:8000/.well-known/agent-card.json

# Send a message via JSON-RPC
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "test-1",
        "role": "user",
        "parts": [{"text": "Hello, agent!"}]
      }
    },
    "id": 1
  }'
```

### 2. Advanced Data Processor Agent (`advanced_agent.py`)

A more sophisticated agent demonstrating:
- **Custom validation** - Reject messages that are too long or empty
- **Progress updates** - Send real-time progress to clients
- **Artifacts** - Attach files/data to responses
- **Lifecycle hooks** - on_startup, on_task_start, on_task_complete, on_task_error
- **Error handling** - Custom error messages

**Run it:**
```bash
python examples/advanced_agent.py
```

This agent runs on port 8001 and performs text analysis with sentiment detection.

## Common Patterns

### Environment Variables

All examples support these environment variables:

- `HOST` - Server host (default: "0.0.0.0")
- `PORT` or `AGENT_PORT` - Server port (default: 8000)
- `RELOAD` - Enable auto-reload for development (default: "false")

**Example:**
```bash
PORT=8080 RELOAD=true python examples/simple_agent.py
```

### Programmatic Server Control

Instead of using `agent.serve()`, you can create the app and run it manually:

```python
from health_universe_a2a import A2AAgent, create_app
import uvicorn

class MyAgent(A2AAgent):
    # ... implement methods ...
    pass

agent = MyAgent()
app = create_app(agent)

# Now you have full control over uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### Development vs Production

For development with auto-reload:
```bash
RELOAD=true python examples/simple_agent.py
```

For production with multiple workers:
```python
from health_universe_a2a import create_app
import uvicorn

app = create_app(MyAgent())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

## Next Steps

- Check out the [main README](../README.md) for SDK documentation
- Read the [API reference](../docs/api.md) for detailed method documentation
- See [QUICKSTART.md](../QUICKSTART.md) for more tutorials
