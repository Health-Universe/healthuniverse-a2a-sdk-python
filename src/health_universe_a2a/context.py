"""Context objects passed to agent process_message methods"""

import os
import uuid
from typing import TYPE_CHECKING, Any

from a2a.types import Message, Part, Role, TaskState, TextPart
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from health_universe_a2a.inter_agent import InterAgentClient


class MessageContext(BaseModel):
    """
    Context for StreamingAgent process_message method.

    Provides methods to send progress updates and artifacts during processing.

    Attributes:
        user_id: User ID from request metadata
        thread_id: Thread/conversation ID from request metadata
        file_access_token: Supabase access token for file operations (if file access enabled)
        auth_token: JWT token from original request (for inter-agent calls)
        metadata: Raw metadata from A2A request
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str | None = None
    thread_id: str | None = None
    file_access_token: str | None = None
    auth_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Internal fields (not for external use)
    _updater: Any = None
    _request_context: Any = None
    _cancelled: bool = False

    async def update_progress(
        self, message: str, progress: float | None = None, status: str = "working"
    ) -> None:
        """
        Send a progress update to the UI via A2A protocol.

        Args:
            message: Status message to display
            progress: Progress from 0.0 to 1.0 (optional)
            status: Task status (default: "working")

        Example:
            await context.update_progress("Loading data...", 0.2)
            await context.update_progress("Processing...", 0.5)

        Note:
            Uses TaskUpdater.update_status() with TaskState.working by default.
            The progress value is stored in message metadata.
        """
        if self._updater:
            # Create message with progress metadata
            metadata = {}
            if progress is not None:
                metadata["progress"] = progress

            # Create A2A Message object
            text_part = TextPart(text=message)
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=text_part)],
                metadata=metadata if metadata else None,
            )

            # Map status string to TaskState
            state_map = {
                "working": TaskState.working,
                "completed": TaskState.completed,
                "failed": TaskState.failed,
                "error": TaskState.failed,
                "queued": TaskState.submitted,
            }
            task_state = state_map.get(status, TaskState.working)

            # Update status via TaskUpdater
            await self._updater.update_status(
                state=task_state,
                message=msg,
                final=False,
                metadata=metadata if metadata else None,
            )

    async def add_artifact(
        self,
        name: str,
        content: str,
        data_type: str = "text/plain",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an artifact to the response via A2A protocol.

        Artifacts appear in the UI and can be downloaded/viewed by users.

        Args:
            name: Artifact name (e.g., "Analysis Results")
            content: Artifact content (string or bytes)
            data_type: MIME type (e.g., "application/json", "image/png")
            description: Optional description
            metadata: Optional metadata dict

        Example:
            await context.add_artifact(
                name="Results",
                content=json.dumps({"score": 0.95}),
                data_type="application/json"
            )

        Note:
            Uses TaskUpdater.add_artifact() with A2A Part objects.
        """
        if self._updater:
            # Create artifact metadata
            artifact_metadata = metadata or {}
            if description:
                artifact_metadata["description"] = description
            artifact_metadata["data_type"] = data_type

            # Create A2A Part with content
            text_part = TextPart(text=content, metadata=artifact_metadata)
            parts = [Part(root=text_part)]

            # Add artifact via TaskUpdater
            await self._updater.add_artifact(
                parts=parts,
                name=name,
                metadata=artifact_metadata,
            )

    def is_cancelled(self) -> bool:
        """
        Check if the task was cancelled by the user.

        Use this to gracefully stop processing if the user cancels the request.

        Returns:
            True if task was cancelled, False otherwise

        Example:
            while processing:
                if context.is_cancelled():
                    return "Processing cancelled by user"
                # ... continue processing
        """
        return self._cancelled

    def create_inter_agent_client(
        self,
        agent_identifier: str,
        local_base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> "InterAgentClient":
        """
        Create an InterAgentClient with automatic JWT propagation.

        This is the recommended way to call other agents, as it automatically
        propagates the auth token from the original request.

        Args:
            agent_identifier: Target agent identifier:
                - "/agent-path" for local agents (same pod)
                - "http(s)://..." for direct URLs
                - "agent-name" for registry lookup
            local_base_url: Override local base URL (default: from env or localhost:8501)
            timeout: Request timeout in seconds (default: 30s, increase for slow agents)
            max_retries: Max retry attempts for transient errors (default: 3)

        Returns:
            Configured InterAgentClient instance

        Example:
            async def process_message(self, message: str, context: MessageContext) -> str:
                # Call a local agent in the same pod
                client = context.create_inter_agent_client("/data-processor")
                result = await client.call(message)

                # Call a remote agent with longer timeout
                client = context.create_inter_agent_client(
                    "https://api.example.com/agent",
                    timeout=300.0  # 5 minutes
                )
                result = await client.call_with_data({"query": message})

                return result.text
        """
        from health_universe_a2a.inter_agent import InterAgentClient

        return InterAgentClient(
            agent_identifier=agent_identifier,
            auth_token=self.auth_token,
            local_base_url=local_base_url
            or os.getenv("LOCAL_AGENT_BASE_URL", "http://localhost:8501"),
            timeout=timeout,
            max_retries=max_retries,
        )


class AsyncContext(MessageContext):
    """
    Context for AsyncAgent process_message method.

    Extends MessageContext with job tracking for long-running tasks.

    Attributes:
        job_id: Async job ID from the backend system
        user_id: User ID from request metadata
        thread_id: Thread/conversation ID from request metadata
        file_access_token: Supabase access token for file operations (if file access enabled)
        metadata: Raw metadata from A2A request
    """

    job_id: str = ""

    # Internal fields
    _update_client: Any = None

    async def update_progress(
        self, message: str, progress: float | None = None, status: str = "working"
    ) -> None:
        """
        Send a progress update (POSTed to backend).

        Updates are stored in the database and displayed to users even if they're offline.

        Args:
            message: Status message
            progress: Progress from 0.0 to 1.0 (optional)
            status: Task status (default: "working")

        Example:
            for i, batch in enumerate(batches):
                await context.update_progress(
                    f"Processing batch {i+1}/{len(batches)}",
                    progress=(i+1)/len(batches)
                )
        """
        if self._update_client:
            await self._update_client.post_update(
                update_type="progress",
                status_message=message,
                progress=progress,
                task_status=status,
            )

    async def add_artifact(
        self,
        name: str,
        content: str,
        data_type: str = "text/plain",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an artifact (POSTed to backend).

        Args:
            name: Artifact name
            content: Artifact content
            data_type: MIME type
            description: Optional description
            metadata: Optional metadata dict

        Example:
            await context.add_artifact(
                name=f"Batch {i} Results",
                content=json.dumps(results),
                data_type="application/json"
            )
        """
        if self._update_client:
            artifact_data: dict[str, Any] = {
                "name": name,
                "content": content,
                "data_type": data_type,
                "description": description,
            }
            if metadata:
                artifact_data["metadata"] = metadata

            await self._update_client.post_update(
                update_type="artifact", artifact_data=artifact_data
            )
