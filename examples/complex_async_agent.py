"""
Complex Async Agent Example

A batch data processor that demonstrates:
- Validation before processing (with immediate ack via SSE)
- File access with Supabase token
- Batch processing with progress updates (POSTed to backend)
- Multiple artifacts
- Error handling and recovery
- Cancellation checking
"""

import asyncio
import json
from typing import Any

from health_universe_a2a import (
    AsyncAgent,
    AsyncContext,
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)


class BatchDataProcessorAgent(AsyncAgent):
    """
    Complex batch processor demonstrating advanced AsyncAgent features.

    Processes large datasets in batches, with:
    - Pre-validation of job parameters
    - File access for reading data
    - Progress updates for each batch
    - Artifact generation per batch
    - Cancellation support
    """

    def __init__(self) -> None:
        super().__init__()
        self.jobs_processed = 0

    def get_agent_name(self) -> str:
        return "Batch Data Processor Pro"

    def get_agent_description(self) -> str:
        return "Processes large datasets in batches with comprehensive progress tracking and error recovery"

    def get_agent_version(self) -> str:
        return "3.0.0"

    def requires_file_access(self) -> bool:
        """Enable file access to read user datasets."""
        return True

    def get_max_duration_seconds(self) -> int:
        """Allow up to 2 hours for batch processing."""
        return 7200

    async def validate_message(self, message: str, metadata: dict[str, Any]) -> ValidationResult:
        """
        Validate job parameters before processing.

        Checks:
        - Valid JSON format
        - Required parameters present
        - Batch size within limits
        - Estimated duration

        Note: Validation happens before processing starts. If accepted,
        an ack is sent via SSE, then processing begins immediately with
        updates POSTed to the backend.
        """
        try:
            request = json.loads(message)

            # Check required fields
            if "file_uri" not in request:
                return ValidationRejected(reason="Missing required field: file_uri")

            if "batch_size" not in request:
                return ValidationRejected(reason="Missing required field: batch_size")

            # Validate batch size
            batch_size = request["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                return ValidationRejected(reason="batch_size must be a positive integer")

            if batch_size > 1000:
                return ValidationRejected(reason="batch_size too large (max 1000 per batch)")

            # Estimate processing time
            # Assume: 10 seconds per batch, estimate total batches
            estimated_batches = request.get("estimated_rows", 10000) // batch_size
            estimated_seconds = estimated_batches * 10

            if estimated_seconds > self.get_max_duration_seconds():
                return ValidationRejected(
                    reason=f"Estimated processing time ({estimated_seconds}s) exceeds maximum ({self.get_max_duration_seconds()}s)"
                )

            self.logger.info(
                f"Job validated: {estimated_batches} batches, ~{estimated_seconds}s"
            )

            return ValidationAccepted(estimated_duration_seconds=estimated_seconds)

        except json.JSONDecodeError:
            return ValidationRejected(reason="Invalid JSON format")
        except Exception as e:
            return ValidationRejected(reason=f"Validation error: {e}")

    async def on_startup(self) -> None:
        """Initialize processor."""
        self.logger.info("Batch Data Processor starting up...")
        self.jobs_processed = 0

    async def on_task_start(self, message: str, context: AsyncContext) -> None:
        """Log job start."""
        self.jobs_processed += 1
        self.logger.info(
            f"Starting batch job #{self.jobs_processed} (job_id: {context.job_id})"
        )

    async def on_task_complete(
        self, message: str, result: str, context: AsyncContext
    ) -> None:
        """Log job completion."""
        self.logger.info(
            f"Batch job #{self.jobs_processed} completed (job_id: {context.job_id})"
        )

    async def on_task_error(
        self, message: str, error: Exception, context: AsyncContext
    ) -> str | None:
        """Handle job errors with recovery suggestions."""
        self.logger.error(f"Batch job #{self.jobs_processed} failed: {error}")

        if isinstance(error, asyncio.TimeoutError):
            return "Job timed out. Consider using smaller batch sizes or fewer batches."

        if isinstance(error, MemoryError):
            return "Out of memory. Try reducing batch_size parameter."

        # Return None to use default error message
        return None

    async def process_message(self, message: str, context: AsyncContext) -> str:
        """
        Process dataset in batches.

        Message format:
        {
            "file_uri": "s3://bucket/dataset.csv",
            "batch_size": 100,
            "operation": "transform" | "aggregate" | "analyze"
        }
        """
        # Parse request
        await context.update_progress("Initializing batch processor...", 0.0)
        request = json.loads(message)

        file_uri = request["file_uri"]
        batch_size = request["batch_size"]
        operation = request.get("operation", "analyze")

        # Simulate loading file metadata
        await context.update_progress(f"Loading file metadata: {file_uri}...", 0.05)
        await asyncio.sleep(1)

        # Simulate determining total rows (would actually read file in real implementation)
        total_rows = request.get("estimated_rows", 5000)
        total_batches = (total_rows + batch_size - 1) // batch_size

        self.logger.info(
            f"Processing {total_rows} rows in {total_batches} batches of {batch_size}"
        )

        await context.update_progress(
            f"Starting batch processing ({total_batches} batches)...", 0.1
        )

        # Process batches
        processed_rows = 0
        batch_results = []

        for batch_num in range(total_batches):
            # Check for cancellation
            if context.is_cancelled():
                await context.update_progress("Cancellation requested, stopping...", None)
                return f"Job cancelled after processing {batch_num} of {total_batches} batches"

            # Calculate progress
            batch_progress = 0.1 + (0.8 * (batch_num / total_batches))

            await context.update_progress(
                f"Processing batch {batch_num + 1}/{total_batches} ({operation})...",
                batch_progress,
            )

            # Simulate batch processing
            await asyncio.sleep(0.5)  # Simulate work

            # Simulate batch result
            batch_result = {
                "batch_number": batch_num + 1,
                "rows_processed": min(batch_size, total_rows - processed_rows),
                "operation": operation,
                "status": "success",
                "metrics": {
                    "records_valid": min(batch_size, total_rows - processed_rows) - 2,
                    "records_invalid": 2,
                    "processing_time_ms": 500,
                },
            }

            batch_results.append(batch_result)
            processed_rows += batch_result["rows_processed"]

            # Create artifact every 10 batches or on last batch
            if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                await context.add_artifact(
                    name=f"Batch Results {batch_num - 8}-{batch_num + 1}",
                    content=json.dumps(batch_results[-10:], indent=2),
                    data_type="application/json",
                    description=f"Processing results for batches {max(1, batch_num - 8)} to {batch_num + 1}",
                )

        # Generate final summary
        await context.update_progress("Generating final summary...", 0.95)

        total_valid = sum(b["metrics"]["records_valid"] for b in batch_results)
        total_invalid = sum(b["metrics"]["records_invalid"] for b in batch_results)
        avg_time = sum(b["metrics"]["processing_time_ms"] for b in batch_results) / len(
            batch_results
        )

        summary = {
            "job_id": context.job_id,
            "file_uri": file_uri,
            "operation": operation,
            "total_rows": total_rows,
            "total_batches": total_batches,
            "batch_size": batch_size,
            "records_valid": total_valid,
            "records_invalid": total_invalid,
            "average_batch_time_ms": avg_time,
            "success": True,
        }

        await context.add_artifact(
            name="Final Summary",
            content=json.dumps(summary, indent=2),
            data_type="application/json",
            description="Comprehensive summary of batch processing job",
        )

        # Create human-readable report
        report = f"""# Batch Processing Report

## Job Details
- **Job ID**: {context.job_id}
- **File**: {file_uri}
- **Operation**: {operation}

## Processing Summary
- **Total Rows**: {total_rows:,}
- **Batches Processed**: {total_batches}
- **Batch Size**: {batch_size}

## Results
- **Valid Records**: {total_valid:,} ({100*total_valid/total_rows:.1f}%)
- **Invalid Records**: {total_invalid:,} ({100*total_invalid/total_rows:.1f}%)
- **Average Batch Time**: {avg_time:.1f}ms

## Status
✅ Job completed successfully!

All batch results have been saved as artifacts. See attached JSON files for detailed per-batch metrics.
"""

        await context.add_artifact(
            name="Processing Report",
            content=report,
            data_type="text/markdown",
            description="Human-readable processing report",
        )

        return f"Successfully processed {total_rows:,} rows in {total_batches} batches. {total_valid:,} valid, {total_invalid:,} invalid records."


# Example usage
if __name__ == "__main__":
    print("Complex Batch Data Processor Agent Example")
    print("\nFeatures demonstrated:")
    print("  - Validation before processing (with estimated duration)")
    print("  - Dual-updater pattern (SSE for ack, POST for processing)")
    print("  - File access with Supabase token")
    print("  - Batch processing with detailed progress")
    print("  - Multiple artifacts (per-batch + summary)")
    print("  - Cancellation checking")
    print("  - Error handling with custom messages")
    print("  - Lifecycle hooks")
    print("\nExample request:")
    print(
        json.dumps(
            {"file_uri": "s3://bucket/large-dataset.csv", "batch_size": 100, "operation": "analyze"},
            indent=2,
        )
    )
    print("\nProcessing flow:")
    print("  1. Validate parameters (batch size, file URI) via validate_message()")
    print("  2. Estimate duration and return ValidationAccepted/Rejected")
    print("  3. Send 'submitted' ack via SSE (immediate_updater)")
    print("  4. Immediately start processing (not enqueued!)")
    print("  5. POST progress updates every batch (background_updater)")
    print("  6. Generate artifacts every 10 batches (POSTed)")
    print("  7. Create final summary and report (POSTed)")
    print("\nKey: AsyncAgent processes immediately after ack, updates via POST")
