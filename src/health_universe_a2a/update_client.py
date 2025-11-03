"""Client for posting background task updates to the Health Universe backend"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BackgroundUpdateClient:
    """
    Client for POSTing updates to the Health Universe backend.

    Used internally by AsyncAgent to send progress updates
    and artifacts during long-running task processing.
    """

    def __init__(self, job_id: str, api_key: str, base_url: str):
        """
        Initialize update client.

        Args:
            job_id: Background job ID
            api_key: API key for authentication
            base_url: Base URL of the Health Universe backend
        """
        self.job_id = job_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=10.0)

    async def post_update(
        self,
        update_type: str,
        progress: float | None = None,
        task_status: str | None = None,
        status_message: str | None = None,
        artifact_data: dict[str, Any] | None = None,
    ) -> None:
        """
        POST an update to the backend.

        Args:
            update_type: Type of update ("progress", "status", "artifact", "log")
            progress: Progress value 0.0-1.0
            task_status: Task status (e.g., "working", "completed")
            status_message: Status message
            artifact_data: Artifact data dict
        """
        endpoint = f"{self.base_url}/a2a/task-updates"

        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "update_type": update_type,
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
                endpoint,
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
        POST final completion status.

        Args:
            message: Final completion message
        """
        await self.post_update(
            update_type="status",
            task_status="completed",
            status_message=message,
            progress=1.0,
        )

    async def post_failure(self, error: str) -> None:
        """
        POST failure status.

        Args:
            error: Error message
        """
        await self.post_update(
            update_type="status",
            task_status="failed",
            status_message=f"Task failed: {error}",
        )

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.client.aclose()
