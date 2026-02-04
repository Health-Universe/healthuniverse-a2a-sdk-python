# Quick Start Guide

Get started with the Health Universe A2A SDK in 5 minutes!

## Installation

Install directly from the public GitHub repository:

```bash
uv pip install git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
```

> **Note:** Using [uv](https://github.com/astral-sh/uv) is recommended for faster dependency management. You can also use `pip install git+...` if preferred.

## Your First Agent

Create a file `my_agent.py`:

```python
from health_universe_a2a import Agent, AgentContext

class GreetingAgent(Agent):
    """A simple greeting agent."""

    def get_agent_name(self) -> str:
        return "Greeting Agent"

    def get_agent_description(self) -> str:
        return "Greets users by name"

    async def process_message(self, message: str, context: AgentContext) -> str:
        # Extract name from message
        name = message.strip()

        # Send progress update
        await context.update_progress("Generating greeting...", 0.5)

        # Generate greeting
        greeting = f"Hello, {name}! Welcome to Health Universe!"

        # Return final message
        return greeting

if __name__ == "__main__":
    GreetingAgent().serve()
```

That's it! You've created your first A2A agent.

## Running Your Agent

```bash
uv run python my_agent.py
```

Your agent is now running at `http://localhost:8000`.

### Test the Agent Card

```bash
curl http://localhost:8000/.well-known/agent.json
```

### Send a Message

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "Alice"}]}}}'
```

## Next Steps

### Add Validation

Validate input before processing:

```python
from health_universe_a2a import Agent, AgentContext, ValidationAccepted, ValidationRejected

class ValidatedAgent(Agent):
    def get_agent_name(self) -> str:
        return "Validated Agent"

    def get_agent_description(self) -> str:
        return "An agent with input validation"

    async def validate_message(
        self, message: str, metadata: dict
    ) -> ValidationAccepted | ValidationRejected:
        if not message.strip():
            return ValidationRejected(reason="Name cannot be empty")

        if len(message) > 50:
            return ValidationRejected(reason="Name too long (max 50 chars)")

        return ValidationAccepted(estimated_duration_seconds=30)

    async def process_message(self, message: str, context: AgentContext) -> str:
        return f"Hello, {message.strip()}!"
```

### Add Artifacts

Generate downloadable results:

```python
import json

async def process_message(self, message: str, context: AgentContext) -> str:
    name = message.strip()

    # Generate greeting
    greeting = f"Hello, {name}!"

    # Create artifact
    await context.add_artifact(
        name="Greeting Card",
        content=json.dumps({
            "greeting": greeting,
            "recipient": name
        }),
        data_type="application/json"
    )

    return greeting
```

### Work with Documents

Read and write documents in the thread:

```python
async def process_message(self, message: str, context: AgentContext) -> str:
    # List all documents in the thread
    documents = await context.document_client.list_documents()

    # Download a document
    if documents:
        content = await context.document_client.download_text(documents[0].id)

    # Write a new document
    await context.document_client.write(
        name="Analysis Results",
        content='{"status": "complete"}',
        filename="results.json"
    )

    return f"Processed {len(documents)} documents"
```

## Common Patterns

### Progress Updates

```python
from health_universe_a2a import UpdateImportance

# Simple progress
await context.update_progress("Working...", 0.5)

# Important milestone (pushed to Navigator UI)
await context.update_progress(
    "Analysis complete!",
    1.0,
    importance=UpdateImportance.NOTICE
)
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
async def process_message(self, message: str, context: AgentContext) -> str:
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
async def process_message(self, message: str, context: AgentContext) -> str:
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

- **[simple_agent.py](examples/simple_agent.py)**: Basic echo agent
- **[medical_classifier.py](examples/medical_classifier.py)**: Simple symptom classifier
- **[document_inventory.py](examples/document_inventory.py)**: List and inspect thread documents
- **[protocol_analyzer.py](examples/protocol_analyzer.py)**: Search, download, and analyze documents

## Documentation

Full documentation available at: https://docs.healthuniverse.com/a2a-sdk

## Need Help?

- Read the [full README](README.md)
- Report issues on [GitHub](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/issues)
- Email [support@healthuniverse.com](mailto:support@healthuniverse.com)
