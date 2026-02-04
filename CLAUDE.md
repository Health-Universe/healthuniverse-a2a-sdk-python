# Health Universe A2A SDK for Python

## Project Overview

This SDK provides a batteries-included framework for building A2A (Agent-to-Agent) protocol-compliant agents on the Health Universe platform. It abstracts away protocol complexity while exposing the necessary hooks for advanced customization.

**Key Goals:**
- Minimal required implementation (just 3 core methods: `get_agent_name()`, `get_agent_description()`, `process_message()`)
- Progressive complexity (start simple, add features as needed)
- Type-safe with full mypy strict mode support
- Production-ready for Health Universe platform integration

## Architecture

### Agent Class

All Health Universe agents use the `Agent` class (alias for `AsyncAgent`):

**Agent** - The primary agent class for all use cases
- Dual-phase execution: validation via SSE, processing via POST
- Background job processing that survives browser disconnection
- Progress updates stored in database (queryable for billing, analytics)
- Supports both short and long-running tasks (up to `get_max_duration_seconds()`)

### Context Objects

The primary context for agents is `AgentContext` (alias for `BackgroundContext`):

```
BaseContext (user_id, thread_id, file_access_token, auth_token, metadata)
    └── AgentContext/BackgroundContext (+update_client for POST, +job_id, +loop for sync)
```

**AgentContext** methods:
- `update_progress(message, progress, status, importance)` - Send progress update
- `add_artifact(name, content, data_type)` - Generate downloadable artifact
- `is_cancelled()` - Check if task was cancelled
- `create_inter_agent_client(agent_id)` - Create client for calling other agents
- `document_client` - Access to DocumentClient for file operations
- `update_progress_sync(message, progress, importance)` - Sync wrapper for ThreadPoolExecutor

### Extensions System

Health Universe platform extensions declared via `get_extensions()`:

| Extension | URI | Purpose |
|-----------|-----|---------|
| File Access v2 | `https://healthuniverse.com/ext/file_access/v2` | S3 file access via NestJS API |
| Background Job | `https://healthuniverse.com/ext/background_job/v1` | Background job lifecycle |
| Log Level | `https://healthuniverse.com/ext/log_level/v1` | Update importance levels |

**UpdateImportance levels:**
- `ERROR` - Something went wrong
- `NOTICE` - Important milestone (pushed to Navigator UI)
- `INFO` - Standard progress (default, stored but not pushed)
- `DEBUG` - Diagnostic information

## Development Conventions

### Code Style

- **Type hints required** on all public APIs and function signatures
- **Strict mypy mode** enabled - no `Any` escape hatches without justification
- **Ruff formatting** - run `uv run ruff format src/` before committing
- **100 character line length** maximum
- **Docstrings** on all public classes and methods (Google style)

### Naming Conventions

- Classes: `PascalCase` (e.g., `Agent`, `AgentContext`, `DocumentClient`)
- Functions/methods: `snake_case` (e.g., `process_message`, `update_progress`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `FILE_ACCESS_EXTENSION_URI`)
- Private members: `_leading_underscore` (e.g., `_cancelled`, `_extract_message`)

### File Organization

```
src/health_universe_a2a/
├── __init__.py          # Public API exports
├── base.py              # A2AAgentBase class
├── async_agent.py       # AsyncAgent
├── context.py           # Context objects
├── documents.py         # DocumentClient for file operations
├── inter_agent.py       # InterAgentClient, AgentRegistry
├── server.py            # HTTP server utilities
├── update_client.py     # BackgroundUpdateClient, BackgroundTaskUpdater
├── nest_client.py       # NestJS S3 API client
└── types/
    ├── extensions.py    # Extension URIs, params, helpers
    └── validation.py    # ValidationAccepted, ValidationRejected
```

## Testing Requirements

### Test Structure

- Tests in `tests/` directory, mirroring `src/` structure
- Use `pytest` with `pytest-asyncio` for async tests
- Run tests: `uv run pytest`
- Run with coverage: `uv run pytest --cov=health_universe_a2a`

### Mocking Patterns

**Mock TaskUpdater for streaming tests:**
```python
from unittest.mock import AsyncMock, MagicMock

mock_updater = MagicMock()
mock_updater.update_status = AsyncMock()
mock_updater.add_artifact = AsyncMock()
mock_updater.complete = AsyncMock()
```

**Mock context for agent tests:**
```python
from health_universe_a2a.context import BackgroundContext

context = BackgroundContext(
    user_id="test-user",
    thread_id="test-thread",
    file_access_token=None,
    auth_token=None,
    metadata={},
    update_client=mock_update_client,
    job_id="test-job",
    loop=asyncio.get_event_loop(),
)
```

### Test Categories

1. **Unit tests** - Individual functions/methods in isolation
2. **Integration tests** - Agent with mocked HTTP/SSE layer
3. **E2E tests** - Full request/response cycle with test server

## Key Patterns

### Document Operations

For reading and writing documents in threads:

```python
async def process_message(self, message: str, context: AgentContext) -> str:
    # List all documents in the thread
    docs = await context.document_client.list_documents()

    # Find specific documents by name
    protocols = await context.document_client.filter_by_name("protocol")

    # Download document content
    content = await context.document_client.download_text(doc.id)

    # Write new document to thread
    await context.document_client.write(
        name="Analysis Results",
        content=json.dumps(results),
        filename="results.json"
    )

    return "Done!"
```

### Inter-Agent Communication

```python
# Method 1: Via call_agent (recommended)
response = await self.call_agent("other-agent", message, context)

# Method 2: With structured data (dict/list)
response = await self.call_agent(
    "processor-agent",
    {"query": message, "format": "json"},
    context
)

# Method 3: Via InterAgentClient directly (for more control)
from health_universe_a2a.inter_agent import InterAgentClient

client = InterAgentClient.from_registry("other-agent")
try:
    result = await client.call(message)
finally:
    await client.close()
```

### Background Job Acknowledgment

```python
from health_universe_a2a.types.extensions import ack_background_job_enqueued

# Send immediate acknowledgment before long processing
ack_message = ack_background_job_enqueued(
    job_id=job_id,
    content="Starting analysis...",
    task_id=task.id,
    context_id=context_id,
)
await updater.update_status(TaskState.submitted, ack_message, final=True)

# SSE closes, background processing continues via POST
```

### ThreadPoolExecutor Sync Updates

For CPU-bound work in background threads:

```python
from concurrent.futures import ThreadPoolExecutor

def cpu_intensive_work(context: BackgroundContext, data):
    for i, chunk in enumerate(data):
        # Sync update from thread
        context.update_progress_sync(
            f"Processing chunk {i+1}/{len(data)}",
            progress=i/len(data),
            importance=UpdateImportance.INFO
        )
        process_chunk(chunk)

# In async process_message
with ThreadPoolExecutor(max_workers=4) as executor:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor,
        cpu_intensive_work,
        context,
        data
    )
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HU_APP_URL` | Agent base URL for agent card | `http://localhost:8000` |
| `HU_NESTJS_URL` | NestJS API URL for document operations | `https://api.healthuniverse.com` |
| `AGENT_REGISTRY_PATH` | Path to agents.json registry | None |
| `BACKGROUND_UPDATE_URL` | URL for POST updates | `https://api.healthuniverse.com` |
| `JOB_STATUS_UPDATE_URL` | URL for job status webhooks | None |
| `JOB_RESULTS_URL` | URL for job results webhooks | None |

## Common Pitfalls

### 1. Forgetting `await` on async methods

```python
# WRONG - silently does nothing
context.update_progress("Working...")

# CORRECT
await context.update_progress("Working...")
```

### 2. Using async methods from ThreadPoolExecutor

```python
# WRONG - will fail or hang
def sync_worker(context):
    await context.update_progress("...")  # Can't await in sync function

# CORRECT - use sync wrapper
def sync_worker(context):
    context.update_progress_sync("...", importance=UpdateImportance.INFO)
```

### 3. Not closing InterAgentClient

```python
# WRONG - connection leak
client = InterAgentClient(url)
result = await client.call(message)
# Client never closed!

# CORRECT - use try/finally or context manager
client = InterAgentClient(url)
try:
    result = await client.call(message)
finally:
    await client.close()
```

### 4. Ignoring cancellation in long loops

```python
# WRONG - task runs forever even if cancelled
for item in huge_list:
    await process(item)

# CORRECT - check cancellation periodically
for item in huge_list:
    if context.is_cancelled():
        return "Task cancelled"
    await process(item)
```

### 5. Not handling ValidationRejected properly

```python
# WRONG - crashes if validation returns rejected
async def handle_request(self, message, context, metadata):
    result = await self.validate_message(message, metadata)
    # result could be ValidationRejected!
    await self.process_message(message, context)

# CORRECT - check validation result type
async def handle_request(self, message, context, metadata):
    result = await self.validate_message(message, metadata)
    if isinstance(result, ValidationRejected):
        await context.updater.reject(reason=result.reason)
        return None
    # Now safe to process
```

## Quick Reference

### Minimal Agent Implementation

```python
from health_universe_a2a import Agent, AgentContext

class MyAgent(Agent):
    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "Does something useful"

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Processing...", 0.5)
        result = await self.do_work(message)
        return f"Result: {result}"

if __name__ == "__main__":
    MyAgent().serve()
```

### Running the Agent

```bash
# Development
uv run python -m my_agent

# With environment
HU_APP_URL=http://localhost:8000 uv run python -m my_agent

# Production
uvicorn my_agent:app --host 0.0.0.0 --port 8000
```

### Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format src/

# Type check
uv run mypy src/

# Build package
uv build
```
