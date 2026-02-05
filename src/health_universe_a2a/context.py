"""Context objects passed to agent process_message methods.

The AgentContext (alias for BackgroundContext) is the primary context class for agents.
It provides access to document operations, progress updates, and inter-agent communication.

Example:
    async def process_message(self, message: str, context: AgentContext) -> str:
        # List documents in the thread
        docs = await context.document_client.list_documents()

        # Update progress
        await context.update_progress("Processing...", 0.5)

        # Write output document
        await context.document_client.write("Results", json.dumps(results))

        return "Done!"
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import RequestContext as _RequestContext
from a2a.server.tasks import TaskUpdater as _TaskUpdater
from pydantic import BaseModel, ConfigDict, Field

from health_universe_a2a.types.extensions import (
    UpdateImportance,
)
from health_universe_a2a.update_client import BackgroundUpdateClient

if TYPE_CHECKING:
    from health_universe_a2a.documents import DocumentClient
    from health_universe_a2a.inspect_ai.logger import InspectLogger
    from health_universe_a2a.inter_agent import InterAgentClient

logger = logging.getLogger(__name__)


class BaseContext(BaseModel):
    """
    Base context class with common fields for all agent contexts.

    Attributes:
        user_id: User ID from request metadata (optional, may not be present)
        thread_id: Thread/conversation ID from request metadata (optional)
        file_access_token: Access token for file operations via NestJS API (optional)
        auth_token: JWT token from original request for inter-agent calls (optional)
        metadata: Raw metadata from A2A request
        extensions: List of extension URIs from the request message (optional)

    Example:
        async def process_message(self, message: str, context: AgentContext) -> str:
            # Access user info
            print(f"User: {context.user_id}")
            print(f"Thread: {context.thread_id}")

            # Check if file access is available
            if context.file_access_token:
                docs = await context.document_client.list_documents()
            return "Done"
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str | None = None
    thread_id: str | None = None
    file_access_token: str | None = None
    auth_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: list[str] | None = None
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
            async def process_message(self, message: str, context: AgentContext) -> str:
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

        # Get inspect_logger if available (BackgroundContext has it, BaseContext doesn't)
        inspect_logger = getattr(self, "inspect_logger", None)

        return InterAgentClient(
            agent_identifier=agent_identifier,
            auth_token=self.auth_token,
            local_base_url=local_base_url
            or os.getenv("LOCAL_AGENT_BASE_URL", "http://localhost:8501"),
            timeout=timeout,
            max_retries=max_retries,
            inspect_logger=inspect_logger,
        )


class BackgroundContext(BaseContext):
    """
    Context for Agent (AsyncAgent) background job execution.

    This is the primary context class for Health Universe agents. It provides:
    - Progress updates that are stored in the database
    - Document operations via the `document_client` property
    - Inter-agent communication via `create_inter_agent_client()`

    Attributes:
        update_client: BackgroundUpdateClient for POSTing updates (always present)
        job_id: Background job ID from the backend system (always present)
        loop: Event loop for sync updates from ThreadPoolExecutor (optional)
        user_id: User ID from request metadata (optional)
        thread_id: Thread/conversation ID from request metadata (optional)
        file_access_token: Access token for file operations (optional)
        metadata: Raw metadata from A2A request
        extensions: List of extension URIs from the request message (optional)

    Example:
        async def process_message(self, message: str, context: AgentContext) -> str:
            # List documents in the thread
            docs = await context.document_client.list_documents()

            # Process with progress updates
            for i in range(10):
                await context.update_progress(
                    f"Processing batch {i+1}/10",
                    progress=(i+1)/10
                )
                await process_batch(i)

            # Write results as a new document
            await context.document_client.write(
                "Final Results",
                json.dumps(results),
                filename="results.json"
            )

            return "All batches processed!"
    """

    update_client: BackgroundUpdateClient
    job_id: str
    loop: Any | None = None  # asyncio event loop for sync updates
    inspect_logger: "InspectLogger | None" = None  # Inspect AI logger for observability
    _documents: "DocumentClient | None" = None

    @property
    def document_client(self) -> "DocumentClient":
        """
        Get document client for file operations in this thread.

        The DocumentClient provides methods to list, read, write, and search
        documents in the current thread. Documents are stored in S3 via the
        Health Universe NestJS backend.

        Returns:
            DocumentClient configured for this thread

        Example:
            # List all documents
            docs = await context.document_client.list_documents()

            # Read a document
            content = await context.document_client.download_text(doc_id)

            # Write a new document
            await context.document_client.write(
                "Analysis Results",
                json.dumps({"score": 0.95}),
                filename="results.json"
            )

            # Filter documents by name
            matches = await context.document_client.filter_by_name("protocol")
        """
        if self._documents is None:
            from health_universe_a2a.documents import DocumentClient

            self._documents = DocumentClient(
                base_url=os.getenv("HU_NESTJS_URL", "https://api.healthuniverse.com"),
                access_token=self.file_access_token or "",
                thread_id=self.thread_id or "",
                inspect_logger=self.inspect_logger,
            )
        return self._documents

    async def update_progress(
        self,
        message: str,
        progress: float | None = None,
        status: str = "working",
        importance: UpdateImportance = UpdateImportance.INFO,
    ) -> None:
        """
        Send a progress update (POSTed to backend).

        Updates are stored in the database and displayed to users even if they're offline.
        Use this to keep users informed of long-running task progress.

        Args:
            message: Status message to display
            progress: Progress from 0.0 to 1.0 (optional)
            status: Task status (default: "working")
            importance: Update importance level (default: INFO).
                Only NOTICE and ERROR are pushed to Navigator UI in real-time.

        Example:
            # Simple progress update
            await context.update_progress("Loading data...", 0.2)

            # Progress with percentage
            for i, batch in enumerate(batches):
                await context.update_progress(
                    f"Processing batch {i+1}/{len(batches)}",
                    progress=(i+1)/len(batches)
                )

            # Important milestone - will show in Navigator UI
            await context.update_progress(
                "Analysis complete!",
                progress=1.0,
                importance=UpdateImportance.NOTICE
            )

        Note:
            Updates are POSTed to the backend and persisted in the database.
        """
        # Log to Inspect AI if logger is available
        if self.inspect_logger:
            self.inspect_logger.log_progress_update(
                message=message,
                progress=progress,
                importance=importance.value if hasattr(importance, "value") else str(importance),
            )

        await self.update_client.post_update(
            update_type="progress",
            status_message=message,
            progress=progress,
            task_status=status,
            importance=importance,
        )

    def update_progress_sync(
        self,
        message: str,
        progress: float | None = None,
        status: str = "working",
        importance: UpdateImportance = UpdateImportance.INFO,
    ) -> None:
        """
        Synchronous wrapper for update_progress.

        Use this when running in a ThreadPoolExecutor or other synchronous context.
        Requires that the `loop` attribute was set when creating the context.

        Args:
            message: Status message
            progress: Progress from 0.0 to 1.0 (optional)
            status: Task status (default: "working")
            importance: Update importance level (default: INFO)

        Example:
            def cpu_intensive_work(context: AgentContext, data):
                for i, chunk in enumerate(data):
                    context.update_progress_sync(
                        f"Processing chunk {i+1}/{len(data)}",
                        progress=i/len(data)
                    )
                    process_chunk(chunk)

            # Call from async code
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor, cpu_intensive_work, context, data
                )

        Note:
            This method schedules the async update on the event loop and
            waits for it to complete (with a 10 second timeout).
        """
        if self.loop is None:
            logger.warning("No event loop available for sync status update")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.update_progress(message, progress, status, importance),
                self.loop,
            )
            # Wait for completion with timeout
            future.result(timeout=10.0)
        except Exception as e:
            logger.error(f"Failed to update progress from thread: {e}")

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

        Artifacts appear in the UI and can be downloaded/viewed by users.
        Use this for intermediate outputs or files that don't need to be
        stored as documents.

        For persistent document storage, use context.document_client.write() instead.

        Args:
            name: Artifact name (e.g., "Analysis Results")
            content: Artifact content as string
            data_type: MIME type (e.g., "application/json", "text/markdown")
            description: Optional description
            metadata: Optional metadata dict

        Example:
            await context.add_artifact(
                name="Batch Results",
                content=json.dumps({"score": 0.95}),
                data_type="application/json",
                description="Results from batch processing"
            )

        Note:
            Artifacts are POSTed to the backend and persisted in the database.
            For file storage in the thread, use context.document_client.write() instead.
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

    def add_artifact_sync(
        self,
        name: str,
        content: str,
        data_type: str = "text/plain",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Synchronous wrapper for add_artifact.

        Use this when running in a ThreadPoolExecutor or other synchronous context.
        Requires that the `loop` attribute was set when creating the context.

        Args:
            name: Artifact name
            content: Artifact content
            data_type: MIME type
            description: Optional description
            metadata: Optional metadata dict
        """
        if self.loop is None:
            logger.warning("No event loop available for sync artifact add")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.add_artifact(name, content, data_type, description, metadata),
                self.loop,
            )
            future.result(timeout=30.0)
        except Exception as e:
            logger.error(f"Failed to add artifact from thread: {e}")


# Alias for simpler API
AgentContext = BackgroundContext


# Internal context for SSE validation phase (used by AsyncAgent internally)
class _SSEContext(BaseContext):
    """
    Internal context for SSE validation phase.

    This is NOT part of the public API. Use AgentContext (BackgroundContext) instead.
    This is only used internally by AsyncAgent during the validation/acknowledgment
    phase before the background task starts.
    """

    updater: _TaskUpdater
    request_context: _RequestContext


# Backwards compatibility alias (internal use only)
StreamingContext = _SSEContext
