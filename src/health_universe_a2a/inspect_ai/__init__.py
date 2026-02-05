"""
Inspect AI Integration for Health Universe A2A SDK.

This module provides Inspect AI integration for A2A agents, enabling:
- Automatic event logging to Inspect AI's .eval format
- Visualization via `inspect view`
- Optional true Inspect AI evaluation with scoring

Quick Start:
    # Automatic logging (enabled by default)
    from health_universe_a2a import Agent, AgentContext

    class MyAgent(Agent):
        async def process_message(self, message: str, context: AgentContext) -> str:
            # SDK operations are automatically logged
            docs = await context.documents.list_documents()

            # Manual logging for custom operations
            context.inspect_logger.log_tool_call("my_tool", {...}, result, duration)
            context.inspect_logger.log_model_call("llm_call", "gpt-4o", messages, response, usage, duration)

            return "Done!"

    # Start agent - logs automatically created in ./inspect_logs
    agent = MyAgent()
    agent.serve()

    # View logs
    # $ inspect view --log-dir ./inspect_logs

Running through Inspect AI eval:
    from health_universe_a2a.inspect_ai import create_agent_task, run_agent_as_eval
    from inspect_ai import eval as inspect_eval

    # Option 1: Using run_agent_as_eval helper
    result = await run_agent_as_eval(agent, message, model="gpt-4o")

    # Option 2: Using create_agent_task + inspect_eval directly
    task = create_agent_task(agent, message, metadata)
    logs = inspect_eval(task, model="gpt-4o", log_dir="./inspect_logs")

Configuration:
    Environment variables:
    - INSPECT_LOG_DIR: Directory for .eval files (default: ./inspect_logs)
    - INSPECT_LOGGING_ENABLED: Enable/disable logging (default: true)
    - INSPECT_EVAL_MODE: Run through inspect_eval (default: true)
    - INSPECT_VIEW_PORT: Port for inspect view subprocess (default: 7575)
    - INSPECT_VIEW_HOST: Host for inspect view (default: 127.0.0.1)
"""

from health_universe_a2a.inspect_ai.integration import (
    create_agent_task,
    inspect_eval_enabled,
    inspect_logging_enabled,
    run_agent_as_eval,
)
from health_universe_a2a.inspect_ai.logger import (
    InspectLogger,
    InspectLoggerAsyncContext,
    InspectLoggerContext,
    get_current_logger,
    set_current_logger,
)
from health_universe_a2a.inspect_ai.schema import (
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
    validate_eval_header,
    validate_eval_sample,
    validate_journal_start,
    validate_summaries,
)
from health_universe_a2a.inspect_ai.viewer import (
    get_log_dir,
    get_viewer_host,
    get_viewer_pid,
    get_viewer_port,
    get_viewer_status,
    get_viewer_url,
    is_viewer_running,
    start_inspect_view,
    stop_inspect_view,
)

__all__ = [
    # Logger
    "InspectLogger",
    "InspectLoggerContext",
    "InspectLoggerAsyncContext",
    "get_current_logger",
    "set_current_logger",
    # Schema (Pydantic models for validation)
    "EvalHeader",
    "EvalSpec",
    "EvalStats",
    "EvalPlan",
    "EvalResults",
    "EvalSample",
    "EvalDataset",
    "SampleSummary",
    "SampleOutput",
    "ModelUsageEntry",
    "JournalStart",
    "validate_eval_header",
    "validate_eval_sample",
    "validate_summaries",
    "validate_journal_start",
    # Viewer
    "start_inspect_view",
    "stop_inspect_view",
    "is_viewer_running",
    "get_viewer_url",
    "get_viewer_status",
    "get_viewer_port",
    "get_viewer_host",
    "get_viewer_pid",
    "get_log_dir",
    # Integration
    "create_agent_task",
    "run_agent_as_eval",
    "inspect_eval_enabled",
    "inspect_logging_enabled",
]
