"""
Inspect AI Integration - Wrap A2A agents in Inspect AI Tasks.

This module provides utilities to run A2A agents through Inspect AI's
evaluation system, enabling true eval/scoring capabilities while
preserving normal agent operation.

The key insight is using asyncio.to_thread() to run inspect_eval() while
passing the original event loop to the solver for cross-loop SDK operations.
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from health_universe_a2a.inspect_ai.logger import InspectLogger, set_current_logger

if TYPE_CHECKING:
    from health_universe_a2a.async_agent import AsyncAgent
    from health_universe_a2a.context import BackgroundContext
    from health_universe_a2a.update_client import BackgroundUpdateClient

logger = logging.getLogger(__name__)


def create_agent_task(
    agent: "AsyncAgent",
    message: str,
    metadata: dict[str, Any] | None = None,
    update_client: "BackgroundUpdateClient | None" = None,
    original_loop: asyncio.AbstractEventLoop | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    job_id: str | None = None,
    file_access_token: str | None = None,
    auth_token: str | None = None,
    scorer: Scorer | None = None,
) -> Task:
    """
    Create an Inspect AI Task that wraps an A2A agent's execution.

    This allows running A2A agents through Inspect AI's evaluation system,
    enabling full observability and optional scoring/evaluation.

    Args:
        agent: The A2A agent to wrap
        message: The message to process
        metadata: Request metadata (optional)
        update_client: BackgroundUpdateClient for POST updates (optional)
        original_loop: Original asyncio event loop for cross-loop operations (optional)
        user_id: User ID from request (optional)
        thread_id: Thread ID from request (optional)
        job_id: Background job ID (optional)
        file_access_token: File access token (optional)
        auth_token: JWT auth token (optional)
        scorer: Optional scorer for evaluation

    Returns:
        Inspect AI Task ready to be passed to inspect_eval()

    Example:
        from health_universe_a2a.inspect_ai import create_agent_task
        from inspect_ai import eval as inspect_eval

        agent = MyAgent()
        task = create_agent_task(
            agent=agent,
            message="Analyze the data",
            metadata={"user_id": "123"},
        )
        results = inspect_eval(task, model="gpt-4o", log_dir="./inspect_logs")
    """
    # Build metadata for the task
    task_metadata = {
        "agent_name": agent.get_agent_name(),
        "agent_type": "a2a",
        **(metadata or {}),
    }
    if user_id:
        task_metadata["user_id"] = user_id
    if thread_id:
        task_metadata["thread_id"] = thread_id
    if job_id:
        task_metadata["job_id"] = job_id

    # Create the solver that wraps agent execution
    agent_solver = _create_agent_solver(
        agent=agent,
        message=message,
        metadata=metadata or {},
        update_client=update_client,
        original_loop=original_loop,
        user_id=user_id,
        thread_id=thread_id,
        job_id=job_id,
        file_access_token=file_access_token,
        auth_token=auth_token,
    )

    # Build the task
    return Task(
        dataset=[
            Sample(
                input=message,
                target="",  # A2A agents don't have predefined targets
                metadata=task_metadata,
            )
        ],
        solver=[agent_solver],
        scorer=scorer,
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.0,  # Deterministic for consistency
        ),
    )


def _create_agent_solver(
    agent: "AsyncAgent",
    message: str,
    metadata: dict[str, Any],
    update_client: "BackgroundUpdateClient | None",
    original_loop: asyncio.AbstractEventLoop | None,
    user_id: str | None,
    thread_id: str | None,
    job_id: str | None,
    file_access_token: str | None,
    auth_token: str | None,
) -> Solver:
    """
    Create a solver that wraps the A2A agent's process_message method.

    The solver runs inside Inspect AI's event loop but uses the original_loop
    for SDK operations via run_coroutine_threadsafe.
    """

    @solver
    def a2a_agent_solver() -> Callable[[TaskState, Generate], Any]:
        """Solver that wraps A2A agent execution."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute the A2A agent's process_message method."""
            logger.info(f"Running A2A agent solver for: {agent.get_agent_name()}")

            # Import here to avoid circular imports
            from health_universe_a2a.context import BackgroundContext

            # Create InspectLogger that writes to TaskState
            inspect_logger = InspectLogger(
                task_name=agent.get_agent_name(),
                model=agent.get_model_name() if hasattr(agent, "get_model_name") else "unknown",
                log_dir=os.getenv("INSPECT_LOG_DIR", "./inspect_logs"),
                input_description=message[:500] if message else "A2A Agent Execution",
            )

            # Set as current logger for SDK operations to find
            set_current_logger(inspect_logger)

            try:
                # Build BackgroundContext for the agent
                # Note: We pass original_loop so sync operations can schedule
                # async calls on the original event loop
                context = BackgroundContext(
                    user_id=user_id,
                    thread_id=thread_id,
                    file_access_token=file_access_token,
                    auth_token=auth_token,
                    metadata=metadata,
                    extensions=None,
                    job_id=job_id or "inspect-eval",
                    update_client=update_client,  # type: ignore[arg-type]
                    loop=original_loop,
                    inspect_logger=inspect_logger,
                )

                # Log task start
                inspect_logger.log_task_state("working", "Starting agent execution")

                # Call lifecycle hooks
                if hasattr(agent, "on_task_start"):
                    await agent.on_task_start(message, context)

                # Run the agent's process_message
                result = await agent.process_message(message, context)

                # Call lifecycle hooks
                if hasattr(agent, "on_task_complete"):
                    await agent.on_task_complete(message, result, context)

                # Store result in state
                state.output.text = result
                state.metadata["agent_result"] = result

                # Log success
                inspect_logger.log_task_state("completed", "Agent execution completed")

                logger.info(f"Agent solver completed successfully")

            except Exception as e:
                logger.error(f"Agent solver failed: {e}", exc_info=True)

                # Call error hook
                if hasattr(agent, "on_task_error"):
                    try:
                        custom_error = await agent.on_task_error(message, e, context)
                        if custom_error:
                            state.output.text = custom_error
                    except Exception:
                        pass

                # Log error
                inspect_logger.log_task_state("failed", str(e))

                # Store error in state
                state.metadata["error"] = str(e)
                if not state.output.text:
                    state.output.text = f"Error: {e}"

            finally:
                # Clear current logger
                set_current_logger(None)

                # Note: We don't finalize here because Inspect AI will
                # capture the events in its own log format

            return state

        return solve

    return a2a_agent_solver()


async def run_agent_as_eval(
    agent: "AsyncAgent",
    message: str,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    log_dir: str | None = None,
    scorer: Scorer | None = None,
    update_client: "BackgroundUpdateClient | None" = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    job_id: str | None = None,
    file_access_token: str | None = None,
    auth_token: str | None = None,
) -> str | None:
    """
    Run an A2A agent through Inspect AI's evaluation system.

    This is a convenience function that creates a task, runs inspect_eval,
    and extracts the result.

    Args:
        agent: The A2A agent to run
        message: The message to process
        metadata: Request metadata (optional)
        model: Model name for Inspect AI (default: agent.get_model_name() or "unknown")
        log_dir: Directory for .eval files (default: INSPECT_LOG_DIR or ./inspect_logs)
        scorer: Optional scorer for evaluation
        update_client: BackgroundUpdateClient for POST updates (optional)
        user_id: User ID from request (optional)
        thread_id: Thread ID from request (optional)
        job_id: Background job ID (optional)
        file_access_token: File access token (optional)
        auth_token: JWT auth token (optional)

    Returns:
        Result string from the agent, or None if evaluation failed

    Example:
        from health_universe_a2a.inspect_ai import run_agent_as_eval

        agent = MyAgent()
        result = await run_agent_as_eval(
            agent=agent,
            message="Analyze the uploaded data",
            model="gpt-4o"
        )
    """
    # Get current event loop to pass to solver
    original_loop = asyncio.get_running_loop()

    # Resolve model name
    eval_model = model
    if not eval_model and hasattr(agent, "get_model_name"):
        eval_model = agent.get_model_name()
    if not eval_model:
        eval_model = "unknown"

    # Resolve log directory
    eval_log_dir = log_dir or os.getenv("INSPECT_LOG_DIR", "./inspect_logs")

    # Create task
    task = create_agent_task(
        agent=agent,
        message=message,
        metadata=metadata,
        update_client=update_client,
        original_loop=original_loop,
        user_id=user_id,
        thread_id=thread_id,
        job_id=job_id,
        file_access_token=file_access_token,
        auth_token=auth_token,
        scorer=scorer,
    )

    logger.info(f"Running agent as eval: {agent.get_agent_name()}")
    logger.info(f"Model: {eval_model}, Log dir: {eval_log_dir}")

    try:
        # Run inspect_eval in a thread to avoid blocking
        # inspect_eval is synchronous but handles async solvers internally
        logs = await asyncio.to_thread(
            lambda: inspect_eval(
                task,
                model=eval_model,
                log_dir=eval_log_dir,
            )
        )

        # Extract result from logs
        if logs and len(logs) > 0 and logs[0].samples:
            sample = logs[0].samples[0]
            if hasattr(sample, "output") and hasattr(sample.output, "text"):
                result = sample.output.text
                logger.info(f"Agent eval completed: {len(result)} chars")
                return result

        logger.warning("No result found in eval logs")
        return None

    except Exception as e:
        logger.error(f"Agent eval failed: {e}", exc_info=True)
        raise


def inspect_eval_enabled() -> bool:
    """
    Check if Inspect eval mode is enabled.

    Returns:
        True if INSPECT_EVAL_MODE is "true" (default), False otherwise
    """
    return os.getenv("INSPECT_EVAL_MODE", "true").lower() == "true"


def inspect_logging_enabled() -> bool:
    """
    Check if Inspect logging is enabled.

    Returns:
        True if INSPECT_LOGGING_ENABLED is "true" (default), False otherwise
    """
    return os.getenv("INSPECT_LOGGING_ENABLED", "true").lower() == "true"
