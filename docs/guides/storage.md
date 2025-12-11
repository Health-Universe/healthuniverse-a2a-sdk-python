# Storage

The SDK provides storage backends for working with files, supporting both local filesystem and S3 via the Health Universe NestJS API.

## Directory Context

The easiest way to work with files is `directory_context`, which downloads all files from S3 to a temporary directory:

```python
from health_universe_a2a import directory_context, AsyncAgent, BackgroundContext

class FileProcessor(AsyncAgent):
    async def run_background(self, query: str, context: BackgroundContext):
        async with directory_context(context.file_access_params) as work_dir:
            # All S3 files are now in work_dir
            input_file = work_dir / "input.csv"

            if input_file.exists():
                data = input_file.read_text()
                # Process data...

                # Write output (automatically uploaded to S3)
                output_file = work_dir / "output.csv"
                output_file.write_text(processed_data)

        return "Files processed!"
```

## Storage Backend

For more control, use the `StorageBackend` interface:

```python
from health_universe_a2a import create_storage_backend, storage_context

# In a streaming agent
async def stream(self, query: str, context: StreamingContext):
    if context.storage:
        # Read uploaded files
        content = context.storage.read_text("input.txt", from_upload=True)

        # Write output files (visible to user)
        context.storage.write_text(
            "output.txt",
            "Results here",
            user_visible=True
        )
```

## Storage Types

### Local Storage

For development and testing:

```python
from health_universe_a2a import LocalStorageBackend

storage = LocalStorageBackend(base_path="/tmp/agent-work")
storage.write_text("test.txt", "Hello")
```

### S3 Storage

For production with Health Universe platform:

```python
from health_universe_a2a import S3StorageBackend, NestJSClient

client = NestJSClient(
    api_url="https://api.healthuniverse.com",
    access_token="your_token"
)

storage = S3StorageBackend(
    client=client,
    thread_id="thread_123",
    work_dir="/tmp/work"
)
```

## Storage Context Manager

Automatically selects the right backend:

```python
from health_universe_a2a import storage_context

with storage_context(
    work_dir="/tmp/work",
    nestjs_token="token",
    thread_id="thread_123",
    nestjs_api_url="https://api.healthuniverse.com"
) as storage:
    # S3StorageBackend is automatically configured
    storage.write_text("output.txt", "data", user_visible=True)
```

Without NestJS credentials, falls back to local storage:

```python
with storage_context(work_dir="/tmp/work") as storage:
    # LocalStorageBackend
    storage.write_text("output.txt", "data")
```
