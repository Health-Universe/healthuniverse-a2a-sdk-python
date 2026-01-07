# Quick Start Guide

Get started with the Health Universe A2A SDK in 5 minutes!

## Installation

```bash
uv pip install health-universe-a2a
```

> **Note:** Using [uv](https://github.com/astral-sh/uv) is recommended for faster dependency management. You can also use `pip` if preferred.

## Your First Agent

Create a file `my_agent.py`:

```python
from health_universe_a2a import StreamingAgent, MessageContext

class GreetingAgent(StreamingAgent):
    """A simple greeting agent."""

    def get_agent_name(self) -> str:
        return "Greeting Agent"

    def get_agent_description(self) -> str:
        return "Greets users by name"

    async def process_message(self, message: str, context: MessageContext) -> str:
        # Extract name from message
        name = message.strip()

        # Send progress update
        await context.update_progress("Generating greeting...", 0.5)

        # Generate greeting
        greeting = f"Hello, {name}! Welcome to Health Universe!"

        # Return final message
        return greeting
```

That's it! You've created your first A2A agent.

## Testing Locally

To test your agent, you'll need to integrate it with an A2A server. Here's a minimal example:

```python
# test_agent.py
import asyncio
from my_agent import GreetingAgent
from health_universe_a2a import MessageContext

async def test():
    agent = GreetingAgent()

    # Create a mock context
    context = MessageContext(user_id="test-user")

    # Test the agent
    result = await agent.process_message("Alice", context)
    print(result)  # Output: Hello, Alice! Welcome to Health Universe!

if __name__ == "__main__":
    asyncio.run(test())
```

## Next Steps

### Add Validation

Validate input before processing:

```python
from health_universe_a2a import ValidationResult

async def validate_message(self, message: str, metadata: dict) -> ValidationResult:
    if not message.strip():
        return ValidationResult(
            accepted=False,
            rejection_reason="Name cannot be empty"
        )

    if len(message) > 50:
        return ValidationResult(
            accepted=False,
            rejection_reason="Name too long (max 50 chars)"
        )

    return ValidationResult(accepted=True)
```

### Add Artifacts

Generate downloadable results:

```python
import json

async def process_message(self, message: str, context: MessageContext) -> str:
    name = message.strip()

    # Generate greeting
    greeting = f"Hello, {name}!"

    # Create artifact
    await context.add_artifact(
        name="Greeting Card",
        content=json.dumps({
            "greeting": greeting,
            "timestamp": "2025-01-01T12:00:00Z"
        }),
        data_type="application/json"
    )

    return greeting
```

### Use Background Processing

For long-running tasks, switch to `AsyncAgent`:

```python
from health_universe_a2a import AsyncAgent, AsyncContext

class DataProcessorAgent(AsyncAgent):
    def get_agent_name(self) -> str:
        return "Data Processor"

    def get_agent_description(self) -> str:
        return "Processes large datasets in background"

    def get_max_duration_seconds(self) -> int:
        return 3600  # 1 hour

    async def process_message(self, message: str, context: AsyncContext) -> str:
        # Process in batches
        for i in range(10):
            await context.update_progress(
                f"Processing batch {i+1}/10",
                progress=(i+1)/10
            )

            # ... do work ...

        return "Processing complete!"
```

## Common Patterns

### Progress Updates

```python
from a2a.types import TaskState

# Simple progress
await context.update_progress("Working...", 0.5)

# With custom status
await context.update_progress("Processing batch 3/10", 0.3, status=TaskState.working)

# Mark as completed
await context.update_progress("Done", 1.0, status=TaskState.completed)
```

### Multiple Artifacts

```python
# Summary artifact
await context.add_artifact(
    name="Summary",
    content=json.dumps(summary_data),
    data_type="application/json"
)

# Report artifact
await context.add_artifact(
    name="Report",
    content=markdown_report,
    data_type="text/markdown"
)
```

### Error Handling

```python
async def process_message(self, message: str, context: MessageContext) -> str:
    try:
        result = await risky_operation(message)
        return f"Success: {result}"
    except ValueError as e:
        # Return user-friendly error
        return f"Invalid input: {e}"
    except Exception as e:
        # Log and return generic error
        self.logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred"
```

### Cancellation

```python
async def process_message(self, message: str, context: MessageContext) -> str:
    items = parse_items(message)

    for i, item in enumerate(items):
        # Check for cancellation
        if context.is_cancelled():
            return f"Cancelled after {i} items"

        await process_item(item)

    return "All items processed!"
```

## Examples

Check out the [examples/](examples/) directory for complete working examples:

- **Simple Realtime**: [simple_streaming_agent.py](examples/simple_streaming_agent.py) - Calculator agent
- **Complex Realtime**: [complex_streaming_agent.py](examples/complex_streaming_agent.py) - Data analyzer
- **Simple Background**: [simple_async_agent.py](examples/simple_async_agent.py) - File processor
- **Complex Background**: [complex_async_agent.py](examples/complex_async_agent.py) - Batch processor

## Documentation

Full documentation available at: https://docs.healthuniverse.com/a2a-sdk

## Need Help?

- 📚 Read the [full README](README.md)
- 💬 Join [Discord](https://discord.gg/healthuniverse)
- 🐛 Report issues on [GitHub](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/issues)
- 📧 Email [support@healthuniverse.com](mailto:support@healthuniverse.com)
