# Background Jobs

All Health Universe agents use `Agent` (alias for `AsyncAgent`) which automatically handles background processing with progress updates via webhooks.

## How It Works

1. Client sends request with webhook URL
2. Agent immediately validates and acknowledges with job ID
3. Agent runs `process_message()` in the background
4. Progress updates sent to webhook during processing
5. Final result sent when complete

## Basic Agent

```python
from health_universe_a2a import Agent, AgentContext

class LongRunningAgent(Agent):
    def get_agent_name(self) -> str:
        return "processor"

    def get_agent_description(self) -> str:
        return "Processes data in the background"

    async def process_message(self, message: str, context: AgentContext) -> str:
        # Send progress updates
        await context.update_progress("Starting...", progress=0.0)

        # Do work
        for i in range(10):
            await some_work()
            await context.update_progress(
                f"Step {i + 1}/10",
                progress=(i + 1) / 10
            )

        return "Processing complete!"
```

## Update Importance Levels

Control how updates are displayed to users:

```python
from health_universe_a2a import UpdateImportance

await context.update_progress(
    "Critical error occurred",
    progress=0.5,
    importance=UpdateImportance.ERROR
)

await context.update_progress(
    "Task completed successfully",
    progress=1.0,
    importance=UpdateImportance.NOTICE
)
```

Available levels:

- `ERROR` - Critical issues
- `NOTICE` - Important information (pushed to Navigator UI)
- `INFO` - General progress (default)
- `DEBUG` - Detailed debugging info

## Adding Artifacts

```python
import json

async def process_message(self, message: str, context: AgentContext) -> str:
    # Generate a result file
    result_data = await process_data()

    await context.add_artifact(
        name="results.json",
        content=json.dumps(result_data),
        data_type="application/json"
    )

    return "Results ready!"
```

## Sync Updates from ThreadPoolExecutor

When running CPU-bound work in a thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from health_universe_a2a import Agent, AgentContext

class CPUIntensiveAgent(Agent):
    def get_agent_name(self) -> str:
        return "cpu-agent"

    def get_agent_description(self) -> str:
        return "Runs CPU-intensive computations"

    async def process_message(self, message: str, context: AgentContext) -> str:
        loop = asyncio.get_running_loop()

        def cpu_intensive_work():
            for i in range(100):
                do_heavy_computation()
                # Use sync method from thread
                context.update_progress_sync(
                    f"Computing {i}%",
                    progress=i / 100
                )

        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, cpu_intensive_work)

        return "Computation complete!"
```

## Cancellation Handling

Check for cancellation in long-running loops:

```python
async def process_message(self, message: str, context: AgentContext) -> str:
    items = parse_items(message)

    for i, item in enumerate(items):
        # Check for cancellation
        if context.is_cancelled():
            return f"Cancelled after {i} items"

        await process_item(item)
        await context.update_progress(f"Processed {i+1}/{len(items)}", i/len(items))

    return "All items processed!"
```

## Configuring Max Duration

Override the default 1-hour timeout:

```python
class LongAgent(Agent):
    def get_agent_name(self) -> str:
        return "long-agent"

    def get_agent_description(self) -> str:
        return "Runs for up to 2 hours"

    def get_max_duration_seconds(self) -> int:
        return 7200  # 2 hours

    async def process_message(self, message: str, context: AgentContext) -> str:
        # Long-running work...
        return "Done!"
```
