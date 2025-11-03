# Health Universe A2A SDK for Python

A simple, batteries-included Python SDK for building [A2A-compliant agents](https://a2a.ai) for the Health Universe platform.

## Features

- 🚀 **Simple API**: Just implement 3 methods to create an agent
- 🔄 **Two Agent Types**: StreamingAgent (streaming) and AsyncAgent (long-running)
- 🎯 **Auto-configuration**: Extensions configured automatically based on requirements
- 📊 **Progress Updates**: Built-in support for progress tracking and artifacts
- ✅ **Validation**: Pre-validate messages before processing
- 🔧 **Lifecycle Hooks**: Customize behavior at key points
- 🏥 **Health Universe Integration**: Works seamlessly with HU platform

## Installation

```bash
uv pip install health-universe-a2a
```

For running agents as HTTP servers:

```bash
uv pip install health-universe-a2a[server]
```

For development:

```bash
uv pip install health-universe-a2a[dev]
```

> **Note:** Using [uv](https://github.com/astral-sh/uv) is recommended for faster, more reliable dependency management. If you don't have uv installed, you can use `pip` instead.

## Quick Start

### Simple Realtime Agent

```python
from health_universe_a2a import StreamingAgent, MessageContext

class CalculatorAgent(StreamingAgent):
    def get_agent_name(self) -> str:
        return "Calculator"

    def get_agent_description(self) -> str:
        return "Performs basic calculations"

    async def process_message(self, message: str, context: MessageContext) -> str:
        await context.update_progress("Calculating...", 0.5)

        # Parse and calculate
        result = eval(message)  # Don't do this in production!

        return f"Result: {result}"
```

### Simple Background Agent

```python
from health_universe_a2a import AsyncAgent, AsyncContext

class FileProcessorAgent(AsyncAgent):
    def get_agent_name(self) -> str:
        return "File Processor"

    def get_agent_description(self) -> str:
        return "Processes files in background"

    def requires_file_access(self) -> bool:
        return True  # Auto-enables file access extension

    async def process_message(self, message: str, context: AsyncContext) -> str:
        await context.update_progress("Loading file...", 0.2)

        # Process file using context.file_access_token
        data = await load_file(context.file_access_token, message)

        await context.update_progress("Processing...", 0.6)
        result = await process_data(data)

        await context.add_artifact(
            name="Results",
            content=json.dumps(result),
            data_type="application/json"
        )

        return "Processing complete!"
```

## Agent Types

### StreamingAgent

Use for **short-running tasks** (< 5 minutes) that stream updates in real-time:

- ✅ User expects immediate feedback
- ✅ Processing takes seconds to minutes
- ✅ Updates streamed via A2A protocol (SSE)
- ❌ Long-running tasks (will timeout)

**Features:**
- Real-time progress updates
- Artifact streaming
- Automatic SSE streaming configuration

### AsyncAgent

Use for **long-running tasks** (hours) that process in the background:

- ✅ Processing takes many minutes to hours
- ✅ User doesn't need to wait for completion
- ✅ Updates POSTed to backend for persistence
- ✅ No timeout constraints

**Features:**
- Pre-validation before job enqueueing
- Progress updates via POST to backend
- Jobs persist across disconnects
- No SSE timeout limits

### Choosing the Right Agent Type

Use this decision tree to select the appropriate agent type:

```
Is the task duration > 5 minutes?
│
├─ NO: Use StreamingAgent
│   └─ User waits for response (chat-like interaction)
│   └─ Real-time feedback via SSE
│   └─ Standards-compliant A2A protocol
│
└─ YES: Use AsyncAgent
    └─ User can close tab and return later
    └─ Progress persists in database
    └─ No timeout constraints
```

**Quick Reference:**

| Question | StreamingAgent | AsyncAgent |
|----------|----------------|------------|
| Task duration? | < 5 minutes | Minutes to hours |
| User interaction? | Interactive (waits) | Async (can leave) |
| Updates mechanism? | SSE streaming | POST to backend |
| Timeout risk? | Yes (~5 min) | No |
| Standards compliant? | Yes (pure A2A) | Custom extension |
| Best for? | Chat responses, quick analysis | Batch processing, large datasets |

**Examples:**
- **StreamingAgent**: Calculator, quick data lookup, real-time analysis
- **AsyncAgent**: Large file processing, training models, batch operations

## Core Concepts

### Message Context

Both agent types receive a `MessageContext` (or `AsyncContext`) with helper methods:

```python
async def process_message(self, message: str, context: MessageContext) -> str:
    # Send progress updates
    await context.update_progress("Working...", 0.5)

    # Add artifacts
    await context.add_artifact(
        name="Results",
        content=data,
        data_type="application/json"
    )

    # Check cancellation
    if context.is_cancelled():
        return "Cancelled by user"

    # Access metadata
    user_id = context.user_id
    thread_id = context.thread_id

    # Access file token (if file access enabled)
    token = context.file_access_token

    return "Done!"
```

### Validation

Validate messages **before** processing (or enqueueing for background jobs):

```python
from health_universe_a2a import ValidationResult

async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
    if len(message) < 10:
        return ValidationResult(
            accepted=False,
            rejection_reason="Message too short (min 10 chars)"
        )

    return ValidationResult(
        accepted=True,
        estimated_duration_seconds=60  # Optional hint
    )
```

**For AsyncAgent**: Validation happens **before** the job is enqueued, preventing invalid jobs from being created.

**For StreamingAgent**: Validation happens before processing starts, immediately rejecting invalid requests.

### Lifecycle Hooks

Customize behavior at key points:

```python
async def on_startup(self) -> None:
    """Called when agent starts up"""
    self.model = await load_model()

async def on_shutdown(self) -> None:
    """Called when agent shuts down"""
    await self.model.unload()

async def on_task_start(self, message: str, context: MessageContext) -> None:
    """Called before processing"""
    self.logger.info(f"Starting task for {context.user_id}")

async def on_task_complete(self, message: str, result: str, context: MessageContext) -> None:
    """Called after successful processing"""
    await self.metrics.increment("tasks_completed")

async def on_task_error(self, message: str, error: Exception, context: MessageContext) -> str | None:
    """Called on error - return custom error message or None for default"""
    if isinstance(error, TimeoutError):
        return "Task timed out. Try a smaller request."
    return None
```

### Configuration Methods

Customize agent behavior:

```python
def get_agent_version(self) -> str:
    """Version string (default: "1.0.0")"""
    return "2.1.0"

def requires_file_access(self) -> bool:
    """Enable file access extension (default: False)"""
    return True

def get_max_duration_seconds(self) -> int:
    """Max duration for AsyncAgent (default: 3600)"""
    return 7200  # 2 hours

def get_supported_input_formats(self) -> list[str]:
    """Supported input MIME types"""
    return ["text/plain", "application/json"]

def get_supported_output_formats(self) -> list[str]:
    """Supported output MIME types"""
    return ["text/plain", "application/json"]
```

## Examples

See the `examples/` directory for complete working examples:

### Realtime Examples
- **[simple_streaming_agent.py](examples/simple_streaming_agent.py)**: Basic calculator showing core concepts
- **[complex_streaming_agent.py](examples/complex_streaming_agent.py)**: Data analyzer with validation, artifacts, and lifecycle hooks

### Background Examples
- **[simple_async_agent.py](examples/simple_async_agent.py)**: File processor showing basic background processing
- **[complex_async_agent.py](examples/complex_async_agent.py)**: Batch processor with validation, batching, and error recovery

## Advanced Usage

### Custom Extensions

If you need extensions beyond file access and background jobs:

```python
from health_universe_a2a.types import AgentExtension

def get_extensions(self) -> list[AgentExtension]:
    extensions = super().get_extensions()  # Get auto-configured extensions

    # Add custom extension
    extensions.append(AgentExtension(
        uri="https://example.com/custom-extension/v1",
        metadata={"some": "config"}
    ))

    return extensions
```

### Error Handling

Control error messages returned to users:

```python
async def on_task_error(self, message: str, error: Exception, context: MessageContext) -> str | None:
    # Log error
    self.logger.error(f"Task failed: {error}", exc_info=True)

    # Return user-friendly error message
    if isinstance(error, ValueError):
        return "Invalid input format. Please check your data."
    elif isinstance(error, TimeoutError):
        return "Request timed out. Try reducing the data size."
    elif isinstance(error, MemoryError):
        return "Out of memory. Please contact support."

    # Return None to use default error message
    return None
```

### Cancellation Support

Check for cancellation during long-running operations:

```python
async def process_message(self, message: str, context: MessageContext) -> str:
    items = parse_items(message)

    for i, item in enumerate(items):
        # Check if user cancelled
        if context.is_cancelled():
            return f"Cancelled after processing {i} of {len(items)} items"

        await process_item(item)
        await context.update_progress(f"Processed {i+1}/{len(items)}", (i+1)/len(items))

    return "All items processed!"
```

## Architecture

### How It Works

1. **Agent Declaration**: Agent declares capabilities via `get_agent_name()`, `get_agent_description()`, etc.
2. **Auto-configuration**: SDK automatically configures extensions based on `requires_file_access()`, agent type, etc.
3. **Validation**: Optional `validate_message()` runs before processing
4. **Processing**: Agent's `process_message()` method does the work
5. **Updates**: Agent calls `context.update_progress()` and `context.add_artifact()` as needed
6. **Completion**: Agent returns final message

### StreamingAgent Flow

```
Request → Validate → Process → Stream Updates → Complete
                                      ↓
                               Via A2A Protocol (SSE)
```

### AsyncAgent Flow

```
Request → Validate → Accept/Reject
                          ↓
                    Enqueue Job
                          ↓
                  Process in Background → POST Updates → Complete
                                               ↓
                                      To Backend Database
```

## Development

### Setup

```bash
# Clone repo
git clone https://github.com/Health-Universe/healthuniverse-a2a-sdk-python
cd healthuniverse-a2a-sdk-python

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Testing

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check src/
uv run mypy src/
```

### Formatting

```bash
uv run ruff format src/
```

## Requirements

- Python 3.10+
- httpx >= 0.27.0
- pydantic >= 2.0.0

## Running Your Agent

### Built-in HTTP Server

The SDK includes a built-in HTTP server for running your agents:

```python
from health_universe_a2a import A2AAgent, MessageContext

class MyAgent(A2AAgent):
    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "Does something useful"

    async def process_message(self, message: str, context: MessageContext) -> str:
        return f"Processed: {message}"

if __name__ == "__main__":
    agent = MyAgent()
    agent.serve()  # Starts server on http://0.0.0.0:8000
```

The server automatically provides:
- **Agent card endpoint**: `GET /.well-known/agent-card.json`
- **JSON-RPC endpoint**: `POST /` (method: "message/send")
- **Health check**: `GET /health`

### Server Configuration

Configure via environment variables or method parameters:

```python
# Via environment variables
# HOST=0.0.0.0 PORT=8080 RELOAD=true python my_agent.py

# Via method parameters
agent.serve(host="0.0.0.0", port=8080, reload=True)
```

### Advanced Server Usage

For more control, use `create_app()`:

```python
from health_universe_a2a import create_app
import uvicorn

agent = MyAgent()
app = create_app(agent)

# Full control over uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

See the [examples/](examples/) directory for complete working examples.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📚 [Documentation](https://docs.healthuniverse.com/a2a-sdk)
- 💬 [Discord Community](https://discord.gg/healthuniverse)
- 🐛 [Issue Tracker](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/issues)
- 📧 [Email Support](mailto:support@healthuniverse.com)

## Related Projects

- [Health Universe Platform](https://healthuniverse.com)
- [A2A Protocol Specification](https://a2a.ai)
- [A2A TypeScript SDK](https://github.com/Health-Universe/a2a-sdk-typescript)
