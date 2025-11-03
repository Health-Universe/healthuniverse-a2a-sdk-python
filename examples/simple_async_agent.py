"""
Simple Async Agent Example

A file processing agent that demonstrates:
- Basic AsyncAgent implementation
- File access extension
- Progress updates via POST
- Simple artifact generation
"""

import asyncio
import json

from health_universe_a2a import AsyncAgent, AsyncContext


class FileProcessorAgent(AsyncAgent):
    """
    Simple file processor that demonstrates AsyncAgent basics.

    Processes a CSV file and generates a summary report.
    """

    def get_agent_name(self) -> str:
        return "Simple File Processor"

    def get_agent_description(self) -> str:
        return "Processes CSV files and generates summary reports in the background"

    def requires_file_access(self) -> bool:
        """Enable file access to read user files."""
        return True

    def get_max_duration_seconds(self) -> int:
        """Allow up to 10 minutes for processing."""
        return 600

    async def process_message(self, message: str, context: AsyncContext) -> str:
        """
        Process a CSV file.

        Expected message format: JSON with 'file_uri' key
        """
        # Parse message
        await context.update_progress("Initializing...", 0.0)

        try:
            request = json.loads(message)
            file_uri = request.get("file_uri")

            if not file_uri:
                return "Error: No file_uri provided in request"

        except json.JSONDecodeError:
            return "Error: Invalid JSON format"

        # Simulate file download
        await context.update_progress(f"Downloading file from {file_uri}...", 0.1)
        await asyncio.sleep(2)  # Simulate download time

        # Simulate file parsing
        await context.update_progress("Parsing CSV file...", 0.3)
        await asyncio.sleep(1)

        # Simulate creating summary (in real implementation, would actually process the file)
        row_count = 1000  # Placeholder
        column_count = 5  # Placeholder

        await context.update_progress("Generating summary...", 0.7)
        await asyncio.sleep(1)

        # Create summary artifact
        summary = {
            "file_uri": file_uri,
            "rows": row_count,
            "columns": column_count,
            "processed_at": "2025-01-01T12:00:00Z",
        }

        await context.add_artifact(
            name="File Summary",
            content=json.dumps(summary, indent=2),
            data_type="application/json",
            description=f"Summary of processed file: {file_uri}",
        )

        await context.update_progress("Complete!", 1.0)

        return f"Successfully processed file with {row_count} rows and {column_count} columns"


# Example usage
if __name__ == "__main__":
    print("Simple File Processor Agent Example")
    print("\nFeatures demonstrated:")
    print("  - AsyncAgent for long-running tasks")
    print("  - File access extension (requires_file_access=True)")
    print("  - Progress updates POSTed to backend")
    print("  - Artifact generation")
    print("\nExample request:")
    print('  {"file_uri": "s3://bucket/data.csv"}')
    print("\nProcessing flow:")
    print("  1. Validate request via validate_message()")
    print("  2. Send 'submitted' ack via SSE")
    print("  3. Process immediately (not enqueued), POST updates")
    print("  4. Generate artifacts and completion status (POSTed)")
    print("\nKey: AsyncAgent processes immediately, uses POST for updates")
