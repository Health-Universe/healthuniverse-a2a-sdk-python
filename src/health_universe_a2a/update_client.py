"""Client for posting background task updates to the Health Universe backend"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
from a2a.server.events import Event, EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    Message,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from health_universe_a2a.types.extensions import (
    HU_LOG_LEVEL_EXTENSION_URI,
    BackgroundTaskResults,
    NavigatorTaskStatus,
    UpdateImportance,
    notify_on_task_completion,
)

# Terminal statuses that signal job completion to Navigator
_TERMINAL_STATUSES = {
    NavigatorTaskStatus.COMPLETED,
    NavigatorTaskStatus.FAILED,
}

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BackgroundUpdateClient:
    """
    Client for POSTing updates to the Health Universe backend.

    Used internally by AsyncAgent to send progress updates
    and artifacts during long-running task processing.

    The callback URLs are passed from the platform at runtime, not hardcoded.
    """

    def __init__(
        self,
        job_id: str,
        api_key: str,
        job_status_update_url: str | None = None,
        job_results_url: str | None = None,
    ):
        """
        Initialize update client.

        Args:
            job_id: Background job ID
            api_key: API key for authentication
            job_status_update_url: URL for intermediate status updates (from platform)
            job_results_url: URL for final job results webhook (from platform)
        """
        self.job_id = job_id
        self.api_key = api_key
        self.job_status_update_url = job_status_update_url
        self.job_results_url = job_results_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def post_update(
        self,
        update_type: str,
        progress: float | None = None,
        task_status: str | None = None,
        status_message: str | None = None,
        artifact_data: dict[str, Any] | None = None,
        importance: UpdateImportance = UpdateImportance.NOTICE,
    ) -> None:
        """
        POST an update to the backend.

        Args:
            update_type: Type of update ("progress", "status", "artifact", "log")
            progress: Progress value 0.0-1.0
            task_status: Task status (e.g., "working", "completed")
            status_message: Status message
            artifact_data: Artifact data dict
            importance: Update importance level (default: NOTICE)
        """
        # Skip if no status update URL provided
        if not self.job_status_update_url:
            logger.warning(f"No job_status_update_url configured - update skipped for job {self.job_id}")
            return

        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "update_type": update_type,
            "importance": importance.value,
        }

        if progress is not None:
            payload["progress"] = progress
        if task_status:
            payload["task_status"] = task_status
        if status_message:
            payload["status_message"] = status_message
        if artifact_data:
            payload["artifact_data"] = artifact_data

        try:
            response = await self.client.post(
                self.job_status_update_url,
                json=payload,
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            logger.debug(f"Posted update for job {self.job_id}: {update_type}")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to POST update for job {self.job_id}: {e}")
            # Don't raise - we don't want update failures to crash the agent
        except Exception as e:
            logger.warning(f"Unexpected error posting update for job {self.job_id}: {e}")

    async def post_completion(self, message: str) -> None:
        """
        POST final completion status to both status update and results URLs.

        This ensures Navigator receives the terminal status and stops the progress bar.

        Args:
            message: Final completion message
        """
        # Always send terminal status update to Navigator (prevents hanging progress bar)
        if self.job_status_update_url:
            status_payload: dict[str, Any] = {
                "job_id": self.job_id,
                "task_status": NavigatorTaskStatus.COMPLETED.value,
                "progress": 1.0,
                "status_message": message,
                "is_terminal": True,
            }
            try:
                response = await self.client.post(
                    self.job_status_update_url,
                    json=status_payload,
                    headers={"X-API-Key": self.api_key},
                )
                response.raise_for_status()
                logger.debug(f"Posted terminal completion status for job {self.job_id}")
            except httpx.HTTPError as e:
                logger.warning(f"Failed to POST completion status for job {self.job_id}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error posting completion status for job {self.job_id}: {e}")

        # Also send to results URL if provided
        if not self.job_results_url:
            logger.debug(f"No job_results_url, skipping results POST for job {self.job_id}")
            return

        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "task_status": NavigatorTaskStatus.COMPLETED.value,
            "status_message": message,
            "progress": 1.0,
        }

        try:
            response = await self.client.post(
                self.job_results_url,
                json=payload,
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            logger.debug(f"Posted completion results for job {self.job_id}")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to POST completion results for job {self.job_id}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error posting completion results for job {self.job_id}: {e}")

    async def post_failure(self, error: str) -> None:
        """
        POST failure status to both status update and results URLs.

        This ensures Navigator receives the terminal status and stops the progress bar.

        Args:
            error: Error message
        """
        error_message = f"Task failed: {error}"

        # Always send terminal status update to Navigator (prevents hanging progress bar)
        if self.job_status_update_url:
            status_payload: dict[str, Any] = {
                "job_id": self.job_id,
                "task_status": NavigatorTaskStatus.FAILED.value,
                "progress": 1.0,
                "status_message": error_message,
                "is_terminal": True,
            }
            try:
                response = await self.client.post(
                    self.job_status_update_url,
                    json=status_payload,
                    headers={"X-API-Key": self.api_key},
                )
                response.raise_for_status()
                logger.debug(f"Posted terminal failure status for job {self.job_id}")
            except httpx.HTTPError as e:
                logger.warning(f"Failed to POST failure status for job {self.job_id}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error posting failure status for job {self.job_id}: {e}")

        # Also send to results URL if provided
        if not self.job_results_url:
            logger.debug(f"No job_results_url, skipping failure results POST for job {self.job_id}")
            return

        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "task_status": NavigatorTaskStatus.FAILED.value,
            "status_message": error_message,
        }

        try:
            response = await self.client.post(
                self.job_results_url,
                json=payload,
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            logger.debug(f"Posted failure results for job {self.job_id}")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to POST failure results for job {self.job_id}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error posting failure results for job {self.job_id}: {e}")

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.client.aclose()


class BackgroundArtifactQueue(EventQueue):
    """
    Event queue that buffers artifacts for batch webhook delivery.

    During background job execution, status updates are logged immediately
    but artifacts are buffered in memory. When the job reaches a terminal
    state, all buffered artifacts are sent to the webhook in a single batch.

    This pattern is required because the Navigator UI expects all artifacts
    to arrive together when the job completes.

    Example:
        queue = BackgroundArtifactQueue(api_key, job_id)
        updater = BackgroundTaskUpdater(queue, task_id, context_id)

        # During processing - artifacts are buffered
        await updater.add_artifact(parts=[TextPart(text="Result 1")])
        await updater.add_artifact(parts=[TextPart(text="Result 2")])

        # On completion - all artifacts sent to webhook
        await updater.update_status(TaskState.completed, message)
        # Calls queue.flush_artifacts() automatically
    """

    def __init__(
        self,
        api_key: str,
        job_id: str,
        job_status_update_url: str | None = None,
        job_results_url: str | None = None,
        max_queue_size: int = 100,
    ) -> None:
        """
        Initialize background artifact queue.

        Args:
            api_key: API key for webhook authentication
            job_id: Background job ID
            job_status_update_url: URL for intermediate status updates (optional)
            job_results_url: URL for final job results webhook (optional)
            max_queue_size: Maximum events to buffer
        """
        super().__init__(max_queue_size=max_queue_size)
        self.job_id = job_id
        self.api_key = api_key
        self.job_status_update_url = job_status_update_url
        self.job_results_url = job_results_url
        self.artifact_buffer: list[TaskArtifactUpdateEvent] = []

    async def enqueue_event(self, event: Event) -> None:
        """
        Process an event - log status updates, buffer artifacts.

        Args:
            event: A2A event (TaskStatusUpdateEvent or TaskArtifactUpdateEvent)
        """
        # Log status updates immediately
        if isinstance(event, TaskStatusUpdateEvent):
            logger.debug(f"[Background Status] {event.status.state}: {event.status.message}")
        else:
            logger.debug(f"[Background Event] {type(event).__name__}")

        # Buffer artifacts to send when processing finishes
        if isinstance(event, TaskArtifactUpdateEvent):
            self.artifact_buffer.append(event)

    async def flush_artifacts(self, state: TaskState) -> None:
        """
        Send all buffered artifacts to the webhook.

        Called automatically by BackgroundTaskUpdater when job reaches
        a terminal state (completed, failed, canceled, rejected).

        Args:
            state: Final task state
        """
        if not self.artifact_buffer:
            logger.debug("No artifacts to flush")
            return

        artifacts = [a.artifact for a in self.artifact_buffer]
        logger.info(f"Flushing {len(artifacts)} artifacts for job {self.job_id}")

        await notify_on_task_completion(
            self.api_key,
            BackgroundTaskResults(
                job_id=self.job_id,
                state=state,
                artifacts=artifacts,
            ),
            url=self.job_results_url,
        )

        self.artifact_buffer.clear()


class BackgroundTaskUpdater(TaskUpdater):
    """
    Enhanced TaskUpdater for background jobs with importance levels and sync support.

    Wraps the base TaskUpdater to add:
    - Importance levels for status updates (ERROR, NOTICE, INFO, DEBUG)
    - Automatic artifact flushing on terminal states
    - Synchronous update_status_sync() for ThreadPoolExecutor contexts
    - POST-based status updates to Navigator platform

    Only NOTICE and ERROR level updates are pushed to the Navigator UI in real-time.
    All updates are stored in the A2A system for debugging and audit purposes.

    Example:
        queue = BackgroundArtifactQueue(api_key, job_id)
        loop = asyncio.get_event_loop()
        updater = BackgroundTaskUpdater(queue, task_id, context_id, loop=loop)

        # From async code
        await updater.update_status(
            TaskState.working,
            message,
            importance=UpdateImportance.NOTICE
        )

        # From sync code (ThreadPoolExecutor)
        updater.update_status_sync(TaskState.working, "Processing...")
    """

    _TERMINAL_STATES = {
        TaskState.completed,
        TaskState.failed,
        TaskState.canceled,
        TaskState.rejected,
    }

    def __init__(
        self,
        event_queue: BackgroundArtifactQueue,
        task_id: str,
        context_id: str,
        loop: Any | None = None,
    ) -> None:
        """
        Initialize background task updater.

        Args:
            event_queue: BackgroundArtifactQueue instance
            task_id: Task ID
            context_id: Context ID
            loop: Event loop for sync updates (optional, required for update_status_sync)
        """
        super().__init__(event_queue=event_queue, task_id=task_id, context_id=context_id)
        self.loop = loop
        self._event_queue = event_queue  # Store typed reference

    def _map_task_state_to_status(self, state: TaskState) -> NavigatorTaskStatus:
        """Map A2A TaskState to Navigator task_status enum."""
        state_mapping: dict[TaskState, NavigatorTaskStatus] = {
            TaskState.working: NavigatorTaskStatus.WORKING,
            TaskState.completed: NavigatorTaskStatus.COMPLETED,
            TaskState.failed: NavigatorTaskStatus.FAILED,
            TaskState.canceled: NavigatorTaskStatus.FAILED,
            TaskState.rejected: NavigatorTaskStatus.FAILED,
        }
        return state_mapping.get(state, NavigatorTaskStatus.WORKING)

    def _extract_message_text(self, message: Message | None) -> str:
        """Extract text content from A2A Message object."""
        if not message:
            return ""

        if message.parts:
            for part in message.parts:
                # Try Part wrapper: Part(root=TextPart(text='...'))
                if part.root is not None and isinstance(part.root, TextPart):
                    return part.root.text

                # Try part as dict with nested structure
                if isinstance(part, dict):
                    if "root" in part and isinstance(part["root"], dict):
                        text_value = part["root"].get("text")
                        if text_value:
                            return str(text_value)
                    text_value = part.get("text")
                    if text_value:
                        return str(text_value)

                # Try direct TextPart (fallback)
                if isinstance(part, TextPart):
                    return part.text

        return ""

    async def _post_status_update(
        self,
        task_status: NavigatorTaskStatus,
        progress: float,
        message: str,
        is_terminal: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post intermediate status update to Health Universe Navigator platform."""
        # Skip posting if no URL is configured
        if not self._event_queue.job_status_update_url:
            return

        payload = {
            "job_id": self._event_queue.job_id,
            "task_status": task_status.value,
            "progress": progress,
            "status_message": message,
            "metadata": metadata or {},
            "is_terminal": is_terminal,
        }

        headers = {"X-API-Key": self._event_queue.api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self._event_queue.job_status_update_url,
                    json=payload,
                    headers=headers,
                )

                if response.status_code != 200:
                    logger.warning(f"Status update failed: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Failed to post status update: {e}")

    async def update_status(
        self,
        state: TaskState,
        message: Message | None = None,
        final: bool = False,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        progress: float = 0.0,
        importance: UpdateImportance = UpdateImportance.NOTICE,
    ) -> None:
        """
        Update task status with importance level.

        Args:
            state: Task state
            message: Optional A2A Message
            final: Whether this is the final update
            timestamp: Optional timestamp
            metadata: Optional metadata dict
            progress: Progress value 0.0-1.0
            importance: Update importance level (default: NOTICE)
        """
        # Add importance to metadata for A2A protocol extension
        metadata = metadata or {}
        metadata[HU_LOG_LEVEL_EXTENSION_URI] = {"importance": importance.value}

        await super().update_status(
            state=state,
            message=message,
            final=final,
            timestamp=timestamp,
            metadata=metadata,
        )

        # Extract message text and map state
        message_text = self._extract_message_text(message)
        task_status = self._map_task_state_to_status(state)
        is_terminal = state in BackgroundTaskUpdater._TERMINAL_STATES

        # Only post updates for NOTICE and ERROR importance (quiet mode)
        # INFO and DEBUG updates are stored in A2A but not pushed to Navigator
        # Terminal states are ALWAYS posted to prevent the progress bar from hanging
        should_post = importance in (UpdateImportance.NOTICE, UpdateImportance.ERROR) or is_terminal

        if should_post:
            await self._post_status_update(
                task_status=task_status,
                progress=progress,
                message=message_text or f"Status: {task_status}",
                is_terminal=is_terminal,
                metadata=metadata,
            )

        # Flush artifacts on terminal states
        if is_terminal:
            await self._event_queue.flush_artifacts(state)

    def update_status_sync(
        self,
        state: TaskState,
        text: str,
        importance: UpdateImportance = UpdateImportance.NOTICE,
        progress: float = 0.0,
    ) -> None:
        """
        Synchronous wrapper for update_status.

        Use this when running in a ThreadPoolExecutor or other synchronous context.
        Requires that the `loop` attribute was set when creating the updater.

        Args:
            state: Task state
            text: Status message text
            importance: Importance level (default: NOTICE)
            progress: Progress value 0.0-1.0
        """
        if self.loop is None:
            logger.warning("No event loop available for sync status update")
            return

        # Create message from text
        text_part = TextPart(text=text)
        message = Message(
            message_id="",  # Will be generated
            role="agent",  # type: ignore[arg-type]
            parts=[Part(root=text_part)],
        )

        # Schedule the coroutine on the original event loop from the thread
        future = asyncio.run_coroutine_threadsafe(
            self.update_status(
                state,
                message,
                importance=importance,
                progress=progress,
            ),
            self.loop,
        )

        # Wait for it to complete (with timeout to avoid hanging)
        try:
            future.result(timeout=10.0)
        except Exception as e:
            logger.error(f"Failed to update status from thread: {e}")

    async def add_artifact(
        self,
        parts: list[Part],
        artifact_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        append: bool | None = None,
        last_chunk: bool | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        """
        Add an artifact (buffered until terminal state).

        Args:
            parts: List of A2A Parts containing artifact content
            artifact_id: Optional artifact ID (auto-generated if not provided)
            name: Optional artifact name
            metadata: Optional metadata dict
            append: Whether to append to existing artifact
            last_chunk: Whether this is the last chunk
            extensions: Optional list of extension URIs
        """
        # Create artifact
        artifact = Artifact(
            artifact_id=artifact_id or f"artifact-{self.task_id}",
            parts=parts,
            name=name,
            metadata=metadata,
        )

        # Create event and enqueue (will be buffered)
        event = TaskArtifactUpdateEvent(
            task_id=self.task_id,
            context_id=self.context_id,
            artifact=artifact,
        )

        await self._event_queue.enqueue_event(event)


def create_background_updater(
    api_key: str,
    job_id: str,
    task_id: str,
    context_id: str,
    loop: Any | None = None,
    job_status_update_url: str | None = None,
    job_results_url: str | None = None,
) -> BackgroundTaskUpdater:
    """
    Factory function to create a BackgroundTaskUpdater with associated queue.

    Args:
        api_key: API key for webhook authentication
        job_id: Background job ID
        task_id: Task ID
        context_id: Context ID
        loop: Event loop for sync updates (optional)
        job_status_update_url: URL for intermediate status updates (from platform)
        job_results_url: URL for final job results webhook (from platform)

    Returns:
        Configured BackgroundTaskUpdater instance
    """
    queue = BackgroundArtifactQueue(
        api_key=api_key,
        job_id=job_id,
        job_status_update_url=job_status_update_url,
        job_results_url=job_results_url,
    )

    return BackgroundTaskUpdater(
        event_queue=queue,
        task_id=task_id,
        context_id=context_id,
        loop=loop,
    )
