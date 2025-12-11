# Streaming Agents

Streaming agents provide real-time responses via Server-Sent Events (SSE).

## Basic Streaming

```python
from health_universe_a2a import StreamingAgent, StreamingContext

class MyAgent(StreamingAgent):
    name = "my-agent"
    description = "A streaming agent"

    async def stream(self, query: str, context: StreamingContext):
        yield "Starting processing..."

        # Simulate work
        for i in range(3):
            yield f"Step {i + 1} complete"

        yield "All done!"
```

Each `yield` sends a text chunk to the client immediately.

## Context Object

The `StreamingContext` provides access to:

```python
async def stream(self, query: str, context: StreamingContext):
    # Access the original message
    print(context.message)

    # Access any extensions
    print(context.extensions)

    # Access storage backend (if configured)
    if context.storage:
        content = context.storage.read_text("input.txt")
```

## Adding Artifacts

Attach files or data to your response:

```python
async def stream(self, query: str, context: StreamingContext):
    yield "Processing..."

    # Add a file artifact
    await context.add_artifact(
        name="result.csv",
        mime_type="text/csv",
        data=b"col1,col2\nval1,val2"
    )

    yield "File attached!"
```

## Error Handling

```python
async def stream(self, query: str, context: StreamingContext):
    try:
        result = await some_operation()
        yield f"Result: {result}"
    except Exception as e:
        yield f"Error: {str(e)}"
        # The framework handles proper error propagation
        raise
```

## Validation

Validate incoming requests before processing:

```python
from health_universe_a2a import ValidationAccepted, ValidationRejected

class ValidatingAgent(StreamingAgent):
    async def validate(self, query: str, context) -> ValidationAccepted | ValidationRejected:
        if len(query) < 10:
            return ValidationRejected(reason="Query too short")
        return ValidationAccepted()

    async def stream(self, query: str, context: StreamingContext):
        # Only called if validation passes
        yield f"Processing: {query}"
```
