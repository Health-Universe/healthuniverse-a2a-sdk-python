"""Context objects passed to agent process_message methods"""

import os
import uuid
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TaskState, TextPart
from pydantic import BaseModel, ConfigDict, Field

from health_universe_a2a.update_client import BackgroundUpdateClient

if TYPE_CHECKING:
    from health_universe_a2a.inter_agent import InterAgentClient


class BaseContext(BaseModel):
    """
    Base context class with common fields for all agent contexts.

    This is an abstract base class - use StreamingContext or BackgroundContext instead.

    Attributes:
        user_id: User ID from request metadata (optional, may not be present)
        thread_id: Thread/conversation ID from request metadata (optional)
        file_access_token: Supabase access token for file operations (optional)
        auth_token: JWT token from original request for inter-agent calls (optional)
        metadata: Raw metadata from A2A request
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str | None = None
    thread_id: str | None = None
    file_access_token: str | None = None
    auth_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    _cancelled: bool = False

    def is_cancelled(self) -> bool:
        """
        Check if the task was cancelled by the user.

        **IMPORTANT: Currently always returns False.**

        The cancellation feature is not fully implemented in this SDK. The
        _cancelled flag is never set to True by the default cancel() method.

        For production use with cancellation support:
        1. Override cancel() in your agent (see A2AAgentBase.cancel() docs)
        2. Implement your own cancellation state tracking
        3. Check that state in your processing loops

        Use this to gracefully stop processing if the user cancels the request.

        Returns:
            True if task was cancelled, False otherwise (currently always False)

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


class StreamingContext(BaseContext):
    """
    Context for StreamingAgent with SSE streaming updates.

    This context is used for short-running tasks (< 5 min) that stream
    progress updates via Server-Sent Events (SSE).

    Attributes:
        updater: TaskUpdater instance for sending SSE updates (always present)
        request_context: Original A2A RequestContext (always present)
        user_id: User ID from request metadata (optional)
        thread_id: Thread/conversation ID from request metadata (optional)
        file_access_token: Supabase access token for file operations (optional)
        auth_token: JWT token from original request for inter-agent calls (optional)
        metadata: Raw metadata from A2A request

    Example:
        async def process_message(self, message: str, context: StreamingContext) -> str:
            await context.update_progress("Loading data...", 0.2)
            data = await load_data(message)

            await context.update_progress("Processing...", 0.6)
            results = await process_data(data)

            await context.add_artifact(
                name="Results",
                content=json.dumps(results),
                data_type="application/json"
            )

            return "Processing complete!"
    """

    updater: TaskUpdater
    request_context: RequestContext

    async def update_progress(
        self, message: str, progress: float | None = None, status: str = "working"
    ) -> None:
        """
        Send a progress update to the UI via A2A protocol (SSE).

        Args:
            message: Status message to display
            progress: Progress from 0.0 to 1.0 (optional)
            status: Task status (default: "working")

        Example:
            await context.update_progress("Loading data...", 0.2)
            await context.update_progress("Processing...", 0.5)

        Note:
            Updates are streamed live via SSE to the client.
        """
        # Create message with progress metadata
        metadata: dict[str, Any] = {}
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
        await self.updater.update_status(
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
        Add an artifact to the response via A2A protocol (SSE).

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
            Artifacts are streamed live via SSE to the client.
        """
        # Create artifact metadata
        artifact_metadata = metadata or {}
        if description:
            artifact_metadata["description"] = description
        artifact_metadata["data_type"] = data_type

        # Create A2A Part with content
        text_part = TextPart(text=content, metadata=artifact_metadata)
        parts = [Part(root=text_part)]

        # Add artifact via TaskUpdater
        await self.updater.add_artifact(
            parts=parts,
            name=name,
            metadata=artifact_metadata,
        )


class BackgroundContext(BaseContext):
    """
    Context for AsyncAgent background job execution.

    This context is used for long-running tasks (> 5 min) that execute
    in the background and post updates to the backend via HTTP.

    Attributes:
        update_client: BackgroundUpdateClient for POSTing updates (always present)
        job_id: Background job ID from the backend system (always present)
        user_id: User ID from request metadata (optional)
        thread_id: Thread/conversation ID from request metadata (optional)
        file_access_token: Supabase access token for file operations (optional)
        metadata: Raw metadata from A2A request

    Example:
        async def process_message(self, message: str, context: BackgroundContext) -> str:
            for i in range(10):
                await context.update_progress(
                    f"Processing batch {i+1}/10",
                    progress=(i+1)/10
                )
                await process_batch(i)

            await context.add_artifact(
                name="Final Results",
                content=json.dumps(results),
                data_type="application/json"
            )

            return "All batches processed!"
    """

    update_client: BackgroundUpdateClient
    job_id: str

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

        Note:
            Updates are POSTed to the backend and persisted in the database.
        """
        await self.update_client.post_update(
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

        Note:
            Artifacts are POSTed to the backend and persisted in the database.
        """
        artifact_data: dict[str, Any] = {
            "name": name,
            "content": content,
            "data_type": data_type,
            "description": description,
        }
        if metadata:
            artifact_data["metadata"] = metadata

        await self.update_client.post_update(update_type="artifact", artifact_data=artifact_data)
