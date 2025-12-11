# Background Jobs

Use `AsyncAgent` for long-running tasks that run in the background and send progress updates via webhooks.

## Basic Background Agent

```python
from health_universe_a2a import AsyncAgent, BackgroundContext

class LongRunningAgent(AsyncAgent):
    name = "processor"
    description = "Processes data in the background"

    async def run_background(self, query: str, context: BackgroundContext):
        # Send progress updates
        await context.update_progress(
            message="Starting...",
            progress=0.0
        )

        # Do work
        for i in range(10):
            await some_work()
            await context.update_progress(
                message=f"Step {i + 1}/10",
                progress=(i + 1) / 10
            )

        return "Processing complete!"
```

## Update Importance Levels

Control how updates are displayed to users:

```python
from health_universe_a2a import UpdateImportance

await context.update_progress(
    message="Critical error occurred",
    progress=0.5,
    importance=UpdateImportance.ERROR
)

await context.update_progress(
    message="Task completed successfully",
    progress=1.0,
    importance=UpdateImportance.NOTICE
)
```

Available levels:

- `ERROR` - Critical issues
- `NOTICE` - Important information
- `INFO` - General progress (default)
- `DEBUG` - Detailed debugging info

## Adding Artifacts

```python
async def run_background(self, query: str, context: BackgroundContext):
    # Generate a result file
    result_data = await process_data()

    await context.add_artifact(
        name="results.json",
        mime_type="application/json",
        data=json.dumps(result_data).encode()
    )

    return "Results ready!"
```

## Sync Updates from ThreadPoolExecutor

When running CPU-bound work in a thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_background(self, query: str, context: BackgroundContext):
    loop = asyncio.get_running_loop()

    def cpu_intensive_work():
        for i in range(100):
            do_heavy_computation()
            # Use sync method from thread
            context.update_progress_sync(
                message=f"Computing {i}%",
                progress=i / 100
            )

    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, cpu_intensive_work)

    return "Computation complete!"
```

## Background Job Extension

The SDK automatically handles the BACKGROUND_JOB extension protocol:

1. Client sends request with webhook URL
2. Agent immediately acknowledges with job ID
3. Agent runs in background, sending updates to webhook
4. Final result sent when complete

No additional configuration needed - just implement `run_background()`.
