"""
InspectLogger - Adapter for logging A2A agent events to Inspect AI format.

This module provides a logger that captures tool calls, model calls, and custom
events during A2A agent execution, then writes them to Inspect AI's .eval format
for viewing in `inspect view`.

Based on the InspectLogger from a2a-grant-reviewer, enhanced for the SDK.
"""

import json
import logging
import os
import time
import uuid as uuid_module
import zipfile
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import JsonValue, ValidationError

from .schema import (
    EvalDataset,
    EvalHeader,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
    JournalStart,
    ModelUsageEntry,
    SampleOutput,
    SampleSummary,
)

logger = logging.getLogger(__name__)

# Context variable to make InspectLogger available to SDK operations
_current_inspect_logger: ContextVar["InspectLogger | None"] = ContextVar(
    "current_inspect_logger", default=None
)


def get_current_logger() -> "InspectLogger | None":
    """Get the current InspectLogger from context (if any)."""
    return _current_inspect_logger.get()


def set_current_logger(inspect_logger: "InspectLogger | None") -> None:
    """Set the current InspectLogger in context."""
    _current_inspect_logger.set(inspect_logger)


class InspectLogger:
    """
    Captures A2A agent events and writes them as Inspect AI logs.

    This logger creates .eval JSON files compatible with `inspect view`,
    allowing full observability into agent execution without running
    actual Inspect AI evaluations.

    Usage:
        logger = InspectLogger(task_name="my_agent")

        # Log a tool call
        logger.log_tool_call(
            function_name="process_document",
            arguments={"filename": "data.csv"},
            result={"rows": 1000},
            duration=2.5
        )

        # Log an LLM call
        logger.log_model_call(
            function_name="analyze",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Analyze..."}],
            response="The analysis shows...",
            usage={"prompt_tokens": 500, "completion_tokens": 200},
            duration=3.2
        )

        # Log custom info
        logger.log_info({"step": "processing", "items": 50})

        # Finalize and write to disk
        log_path = logger.finalize()

    Attributes:
        task_name: Name of the task being logged
        model: Model name used by the agent
        log_dir: Directory for .eval files
        input_description: Description of the task input
    """

    def __init__(
        self,
        task_name: str,
        model: str = "unknown",
        log_dir: str | None = None,
        input_description: str = "A2A Agent Execution",
    ):
        """
        Initialize the InspectLogger.

        Args:
            task_name: Name of the task (e.g., "analysis_agent")
            model: Model name used by the agent (e.g., "gpt-4o")
            log_dir: Directory to write .eval files (default: INSPECT_LOG_DIR or ./inspect_logs)
            input_description: Description of the task input
        """
        self.task_name = task_name
        self.model = model
        self.log_dir = log_dir or os.getenv("INSPECT_LOG_DIR", "./inspect_logs")
        self.input_description = input_description

        # Generate unique IDs
        self.eval_id = str(uuid_module.uuid4())
        self.run_id = str(uuid_module.uuid4())
        self.sample_id = str(uuid_module.uuid4())

        # Track events
        self.events: list[dict[str, Any]] = []

        # Track timing
        self.start_time = datetime.now(timezone.utc)
        self.working_time = 0.0

        # Track spans for grouping events
        self._span_stack: list[str] = []
        self._current_span_id: str | None = None

        # Track usage across all model calls
        self._total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Ensure log directory exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"InspectLogger initialized for task={task_name}, log_dir={self.log_dir}")

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_uuid(self) -> str:
        """Generate a short UUID for events."""
        return str(uuid_module.uuid4())[:8]

    def _serialize_value(self, value: Any) -> JsonValue:
        """Safely serialize a value to JSON-compatible format."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        else:
            return str(value)

    def log_tool_call(
        self,
        function_name: str,
        arguments: dict[str, Any],
        result: Any,
        duration: float,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log a tool-like operation as a ToolEvent.

        Args:
            function_name: Name of the function/tool (e.g., "list_documents")
            arguments: Input arguments to the function
            result: Return value from the function
            duration: Execution time in seconds
            metadata: Optional additional metadata
            error: Error message if the call failed
        """
        timestamp = self._get_timestamp()

        # Serialize result to string if needed
        if isinstance(result, dict):
            result_str = json.dumps(result, default=str)
        elif isinstance(result, str):
            result_str = result
        else:
            result_str = str(result)

        # Truncate very long results
        if len(result_str) > 10000:
            result_str = result_str[:10000] + "... (truncated)"

        event = {
            "event": "tool",
            "type": "function",
            "uuid": self._generate_uuid(),
            "span_id": self._current_span_id,
            "timestamp": timestamp,
            "working_start": self.working_time,
            "id": self._generate_uuid(),
            "function": function_name,
            "arguments": self._serialize_value(arguments),
            "result": result_str,
            "working_time": duration,
            "completed": self._get_timestamp(),
        }

        if metadata:
            event["metadata"] = self._serialize_value(metadata)

        if error:
            event["error"] = {"message": error}
            event["failed"] = True

        self.events.append(event)
        self.working_time += duration

        logger.debug(f"Logged tool call: {function_name} ({duration:.2f}s)")

    def log_model_call(
        self,
        function_name: str,
        model: str,
        messages: list[dict[str, str]],
        response: str,
        usage: dict[str, int] | None = None,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log an LLM API call as an InfoEvent.

        Args:
            function_name: Name of the function making the call (e.g., "analyze_document")
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514")
            messages: List of message dicts with role and content
            response: Model response text
            usage: Token usage dict with prompt_tokens, completion_tokens, etc.
            duration: API call duration in seconds
            metadata: Optional additional metadata
            error: Error message if the call failed
        """
        timestamp = self._get_timestamp()

        # Truncate response for logging
        response_truncated = response[:2000] if len(response) > 2000 else response

        # Create a rich info event for model calls
        data = {
            "type": "model_call",
            "function": function_name,
            "model": model,
            "messages": messages,
            "response": response_truncated,
            "response_length": len(response),
            "duration_seconds": duration,
        }

        if usage:
            data["usage"] = usage
            # Track total usage
            self._total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self._total_usage["total_tokens"] += usage.get("total_tokens", 0)

        if metadata:
            data["metadata"] = metadata

        if error:
            data["error"] = error

        event = {
            "event": "info",
            "uuid": self._generate_uuid(),
            "span_id": self._current_span_id,
            "timestamp": timestamp,
            "working_start": self.working_time,
            "source": f"model:{function_name}",
            "data": self._serialize_value(data),
        }

        self.events.append(event)
        self.working_time += duration

        logger.debug(f"Logged model call: {function_name} via {model} ({duration:.2f}s)")

    def log_info(
        self,
        data: dict[str, Any],
        source: str = "agent",
    ) -> None:
        """
        Log custom metrics or milestones as an InfoEvent.

        Args:
            data: Custom data to log
            source: Source identifier for the event (e.g., "agent", "lifecycle")
        """
        timestamp = self._get_timestamp()

        event = {
            "event": "info",
            "uuid": self._generate_uuid(),
            "span_id": self._current_span_id,
            "timestamp": timestamp,
            "working_start": self.working_time,
            "source": source,
            "data": self._serialize_value(data),
        }

        self.events.append(event)

        logger.debug(f"Logged info event from {source}")

    def log_document_op(
        self,
        operation: str,
        doc_id: str | None = None,
        doc_name: str | None = None,
        size: int | None = None,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log a document operation (list, download, write).

        Args:
            operation: Operation type ("list", "download", "write")
            doc_id: Document ID (if applicable)
            doc_name: Document name (if applicable)
            size: Size in bytes (if applicable)
            duration: Operation duration in seconds
            metadata: Optional additional metadata
            error: Error message if the operation failed
        """
        arguments = {}
        if doc_id:
            arguments["doc_id"] = doc_id
        if doc_name:
            arguments["doc_name"] = doc_name

        result: dict[str, Any] = {"success": error is None}
        if size is not None:
            result["size_bytes"] = size

        self.log_tool_call(
            function_name=f"document_{operation}",
            arguments=arguments,
            result=result,
            duration=duration,
            metadata=metadata,
            error=error,
        )

    def log_inter_agent_call(
        self,
        agent_identifier: str,
        message: str,
        response: Any,
        duration: float,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log an inter-agent call.

        Args:
            agent_identifier: Target agent identifier or URL
            message: Message sent to the agent
            response: Response from the agent
            duration: Call duration in seconds
            metadata: Optional additional metadata
            error: Error message if the call failed
        """
        # Truncate message for logging
        message_truncated = message[:500] if len(message) > 500 else message

        self.log_tool_call(
            function_name="inter_agent_call",
            arguments={
                "agent": agent_identifier,
                "message": message_truncated,
            },
            result=response,
            duration=duration,
            metadata=metadata,
            error=error,
        )

    def log_progress_update(
        self,
        message: str,
        progress: float | None = None,
        importance: str = "INFO",
    ) -> None:
        """
        Log a progress update.

        Args:
            message: Progress message
            progress: Progress value 0.0-1.0 (optional)
            importance: Importance level (INFO, NOTICE, ERROR)
        """
        data: dict[str, Any] = {
            "type": "progress_update",
            "message": message,
            "importance": importance,
        }
        if progress is not None:
            data["progress"] = progress

        self.log_info(data, source="a2a:progress")

    def begin_span(self, name: str, span_type: str | None = None) -> str:
        """
        Start a named span for grouping related events.

        Args:
            name: Name of the span (e.g., "document_processing")
            span_type: Optional type of the span

        Returns:
            The span ID
        """
        span_id = self._generate_uuid()
        parent_id = self._current_span_id

        event = {
            "event": "span_begin",
            "uuid": self._generate_uuid(),
            "span_id": parent_id,  # Parent span
            "timestamp": self._get_timestamp(),
            "working_start": self.working_time,
            "id": span_id,
            "name": name,
            "parent_id": parent_id,
        }

        if span_type:
            event["type"] = span_type

        self.events.append(event)

        # Push span onto stack
        self._span_stack.append(span_id)
        self._current_span_id = span_id

        logger.debug(f"Started span: {name} ({span_id})")
        return span_id

    def end_span(self, span_id: str) -> None:
        """
        End a span.

        Args:
            span_id: The span ID to end
        """
        event = {
            "event": "span_end",
            "uuid": self._generate_uuid(),
            "span_id": self._current_span_id,
            "timestamp": self._get_timestamp(),
            "working_start": self.working_time,
            "id": span_id,
        }

        self.events.append(event)

        # Pop span from stack
        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()
            self._current_span_id = self._span_stack[-1] if self._span_stack else None

        logger.debug(f"Ended span: {span_id}")

    def log_task_state(
        self,
        state: str,
        message: str | None = None,
    ) -> None:
        """
        Log an A2A task state transition.

        Args:
            state: Task state (e.g., "working", "completed", "failed")
            message: Optional status message
        """
        data: dict[str, Any] = {
            "type": "task_state",
            "state": state,
        }
        if message:
            data["message"] = message

        self.log_info(data, source="a2a:task")

    def finalize(
        self,
        success: bool = True,
        error: str | None = None,
        scores: dict[str, Any] | None = None,
        output: str | None = None,
    ) -> str:
        """
        Finalize the log and write to disk.

        Args:
            success: Whether the execution was successful
            error: Error message if not successful
            scores: Optional scores to include
            output: Optional output text

        Returns:
            Path to the written log file
        """
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - self.start_time).total_seconds()

        # Log final state
        self.log_task_state("completed" if success else "failed", error)

        # Build model usage (validated via Pydantic)
        model_usage_entry = ModelUsageEntry(
            input_tokens=self._total_usage.get("input_tokens", 0),
            output_tokens=self._total_usage.get("output_tokens", 0),
            total_tokens=self._total_usage.get("total_tokens", 0),
        )
        model_usage_dict = {self.model: model_usage_entry}

        # Build EvalSample (validated via Pydantic)
        sample_output = SampleOutput(model=self.model, choices=[])
        if output:
            sample_output = SampleOutput(model=self.model, choices=[], text=output)

        sample = EvalSample(
            id=self.sample_id,
            epoch=1,
            input=self.input_description,
            target="",
            messages=[],
            output=sample_output,
            events=self.events,
            model_usage=model_usage_dict,
            metadata={},
            store={},
            started_at=self.start_time.isoformat(),
            completed_at=end_time.isoformat(),
            total_time=total_time,
            working_time=self.working_time,
            uuid=self.sample_id,
            scores=scores,
            error={"message": error} if error else None,
        )

        # Build EvalSpec (validated via Pydantic)
        eval_spec = EvalSpec(
            eval_id=self.eval_id,
            run_id=self.run_id,
            created=self.start_time.isoformat(),
            task=self.task_name,
            task_id=f"{self.task_name}-{self.eval_id[:8]}",
            task_version=1,
            model=self.model,
            dataset=EvalDataset(
                name="a2a_execution",
                samples=1,
                sample_ids=[self.sample_id],
                shuffled=False,
            ),
            packages={"health_universe_a2a": "0.2.0"},
        )

        # Build EvalStats (validated via Pydantic)
        stats = EvalStats(
            started_at=self.start_time.isoformat(),
            completed_at=end_time.isoformat(),
            model_usage=model_usage_dict,
        )

        # Build EvalHeader (validated via Pydantic)
        header = EvalHeader(
            version=2,
            status="success" if success else "error",
            eval=eval_spec,
            plan=EvalPlan(name="a2a_agent", steps=[], config={}),
            results=EvalResults(
                total_samples=1,
                completed_samples=1 if success else 0,
                scores=[],
            ),
            stats=stats,
            error={"message": error} if error else None,
        )

        # Build sample summary (validated via Pydantic)
        sample_summary = SampleSummary(
            id=1,
            epoch=1,
            input=self.input_description,
            target="",
            metadata={},
            scores={},
            model_usage=model_usage_dict,
            started_at=self.start_time.isoformat(),
            completed_at=end_time.isoformat(),
            total_time=total_time,
            working_time=self.working_time,
            uuid=self.sample_id,
            retries=0,
            completed=success,
            message_count=len([e for e in self.events if e.get("event") == "model"]),
        )

        # Build journal start (validated via Pydantic)
        journal_start = JournalStart(
            eval_id=self.eval_id,
            run_id=self.run_id,
            task=self.task_name,
            model=self.model,
            created=self.start_time.isoformat(),
        )

        # Write to .eval file (Zip archive format for inspect view compatibility)
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.task_name}_{timestamp_str}_{self.eval_id[:8]}.eval"
        filepath = os.path.join(self.log_dir, filename)

        # Create Zip archive with Inspect AI structure
        # All data is pre-validated by Pydantic models above
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write header.json
            zf.writestr(
                "header.json",
                header.model_dump_json(indent=2, exclude_none=True),
            )

            # Write sample to samples/1_epoch_1.json
            zf.writestr(
                "samples/1_epoch_1.json",
                sample.model_dump_json(indent=2, exclude_none=True),
            )

            # Write summaries.json (array of sample summaries - required by inspect view)
            summaries = [sample_summary.model_dump()]
            zf.writestr("summaries.json", json.dumps(summaries, indent=2, default=str))

            # Write _journal/start.json (required by inspect view)
            zf.writestr(
                "_journal/start.json",
                journal_start.model_dump_json(indent=2),
            )

            # Write _journal/summaries/1.json (same format as summaries.json)
            zf.writestr("_journal/summaries/1.json", json.dumps(summaries, indent=2, default=str))

        logger.info(f"InspectLogger finalized: {filepath} ({len(self.events)} events)")
        return filepath


class InspectLoggerContext:
    """Context manager for InspectLogger with automatic span management."""

    def __init__(
        self, logger: InspectLogger, span_name: str, span_type: str | None = None
    ):
        self.logger = logger
        self.span_name = span_name
        self.span_type = span_type
        self.span_id: str | None = None

    def __enter__(self) -> "InspectLoggerContext":
        self.span_id = self.logger.begin_span(self.span_name, self.span_type)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if self.span_id:
            self.logger.end_span(self.span_id)
        return False


class InspectLoggerAsyncContext:
    """Async context manager for InspectLogger with automatic span management."""

    def __init__(
        self, logger: InspectLogger, span_name: str, span_type: str | None = None
    ):
        self.logger = logger
        self.span_name = span_name
        self.span_type = span_type
        self.span_id: str | None = None

    async def __aenter__(self) -> "InspectLoggerAsyncContext":
        self.span_id = self.logger.begin_span(self.span_name, self.span_type)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if self.span_id:
            self.logger.end_span(self.span_id)
        return False
