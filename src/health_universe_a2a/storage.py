"""
Storage backend abstractions for Health Universe agents.

Provides unified file access across local filesystem and S3 (via NestJS API).
"""

import hashlib
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from health_universe_a2a.nest_client import NestJSClient
from health_universe_a2a.types.extensions import FileAccessExtensionParams

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """
    Abstract base class for storage operations.

    Provides a unified interface for file operations across different
    storage backends (local filesystem, S3 via NestJS).

    All paths are relative to the backend's working directory.
    """

    @abstractmethod
    def write_text(self, path: str, content: str, user_visible: bool = True) -> None:
        """
        Write text content to a file.

        Args:
            path: Relative path to the file
            content: Text content to write
            user_visible: Whether the file should be visible to users (S3 only)
        """
        pass

    @abstractmethod
    def read_text(self, path: str, from_upload: bool = False) -> str:
        """
        Read text content from a file.

        Args:
            path: Relative path to the file
            from_upload: If True, read from user uploads; if False, from agent outputs

        Returns:
            Text content of the file
        """
        pass

    @abstractmethod
    def exists(self, path: str, from_upload: bool = False) -> bool:
        """
        Check if a file exists.

        Args:
            path: Relative path to the file
            from_upload: If True, check user uploads; if False, agent outputs

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def write_bytes(self, path: str, content: bytes, user_visible: bool = True) -> None:
        """
        Write binary content to a file.

        Args:
            path: Relative path to the file
            content: Binary content to write
            user_visible: Whether the file should be visible to users (S3 only)
        """
        pass

    @abstractmethod
    def read_bytes(self, path: str, from_upload: bool = False) -> bytes:
        """
        Read binary content from a file.

        Args:
            path: Relative path to the file
            from_upload: If True, read from user uploads; if False, from agent outputs

        Returns:
            Binary content of the file
        """
        pass

    @abstractmethod
    def get_path(self, *parts: str) -> str:
        """
        Join path parts and return a path string.

        Args:
            *parts: Path components to join

        Returns:
            Joined path string
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any temporary resources (temp directories, connections, etc.)."""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend with optional namespaced temp directories.

    For local development and testing. Can create isolated temp directories
    per thread_id to prevent conflicts.

    Example:
        # Simple usage
        backend = LocalStorageBackend("output")
        backend.write_text("result.json", '{"data": 123}')

        # With thread isolation
        backend = LocalStorageBackend("output", thread_id="thread_123")
        # Creates /tmp/tmp_abc123_xyz/output/
    """

    def __init__(
        self,
        base_dir: str,
        thread_id: str | None = None,
        use_temp: bool = True,
    ):
        """
        Initialize local storage backend.

        Args:
            base_dir: Base directory name (e.g., 'output', 'work')
            thread_id: Thread ID for namespacing (optional)
            use_temp: If True and thread_id provided, use temp directory
        """
        self.is_temp = False
        self.temp_dir: str | None = None

        if thread_id and use_temp:
            # Create namespaced temp directory
            safe_id = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
            prefix_name = f"tmp_{safe_id}_"
            self.temp_dir = tempfile.mkdtemp(prefix=prefix_name)
            self.base_dir = Path(self.temp_dir) / base_dir
            self.is_temp = True
        else:
            # Use regular directory
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, content: str, user_visible: bool = True) -> None:
        """Write text content to a local file."""
        file_path = self.base_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def read_text(self, path: str, from_upload: bool = False) -> str:
        """Read text content from a local file."""
        file_path = self.base_dir / path
        return file_path.read_text()

    def exists(self, path: str, from_upload: bool = False) -> bool:
        """Check if a local file exists."""
        file_path = self.base_dir / path
        return file_path.exists()

    def write_bytes(self, path: str, content: bytes, user_visible: bool = True) -> None:
        """Write binary content to a local file."""
        file_path = self.base_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    def read_bytes(self, path: str, from_upload: bool = False) -> bytes:
        """Read binary content from a local file."""
        file_path = self.base_dir / path
        return file_path.read_bytes()

    def get_path(self, *parts: str) -> str:
        """Join path parts using OS-appropriate separator."""
        return str(Path(*parts))

    def cleanup(self) -> None:
        """Clean up temporary directory if it was created."""
        if self.is_temp and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

    def get_local_path(self) -> str:
        """Get the actual local filesystem path (useful for debugging)."""
        return str(self.base_dir)


class S3StorageBackend(StorageBackend):
    """
    S3 storage backend using NestJS API and presigned URLs.

    All operations go through the NestJS document storage API which handles
    S3 presigned URLs, versioning, and access control.

    Example:
        backend = S3StorageBackend(
            nestjs_api_url="https://apps.healthuniverse.com/api/v1",
            work_dir="analysis",
            token="jwt_token_here",
            thread_id="thread_123",
        )
        backend.write_text("result.json", '{"data": 123}')
        content = backend.read_text("input.pdf", from_upload=True)
    """

    def __init__(
        self,
        nestjs_api_url: str,
        work_dir: str,
        token: str | SecretStr,
        thread_id: str,
    ):
        """
        Initialize S3 storage backend.

        Args:
            nestjs_api_url: Base URL of the NestJS API
            work_dir: Working directory name (for agent identification)
            token: JWT authentication token (str or SecretStr)
            thread_id: Thread ID for namespacing
        """
        self.nestjs_api_url = nestjs_api_url
        self.work_dir = work_dir
        self.token = token if isinstance(token, SecretStr) else SecretStr(token)
        self.thread_id = thread_id
        self._client: NestJSClient | None = None

        # Cache for document name -> document ID mapping
        self._document_cache: dict[str, dict[str, Any]] = {}
        self._cache_loaded = False
        self._cache_timestamp: float | None = None
        self._cache_ttl_seconds = 60  # Refresh cache after 60 seconds

    def _get_client(self) -> NestJSClient:
        """Get or create NestJS client."""
        if self._client is None:
            self._client = NestJSClient(self.nestjs_api_url, self.token)
        return self._client

    def _is_cache_stale(self) -> bool:
        """Check if cache needs to be refreshed based on TTL."""
        if not self._cache_loaded or self._cache_timestamp is None:
            return True
        elapsed = time.time() - self._cache_timestamp
        return elapsed > self._cache_ttl_seconds

    def _load_document_cache(self, force_refresh: bool = False) -> None:
        """Load all documents in the thread for name->ID mapping."""
        if not force_refresh and self._cache_loaded and not self._is_cache_stale():
            return

        client = self._get_client()
        documents = client.list_documents_sync(self.thread_id)

        self._document_cache = {}
        for doc in documents:
            self._document_cache[doc["name"]] = {
                "id": doc["id"],
                "documentType": doc["documentType"],
                "storagePath": doc.get("storagePath", ""),
            }

        self._cache_loaded = True
        self._cache_timestamp = time.time()

    def _update_cache_entry(
        self,
        document_name: str,
        document_id: str,
        document_type: str,
        storage_path: str,
    ) -> None:
        """Update cache with a new or modified document entry."""
        self._document_cache[document_name] = {
            "id": document_id,
            "documentType": document_type,
            "storagePath": storage_path,
        }

    def _get_document_name(self, path: str) -> str:
        """Extract document name from path (just the filename)."""
        return Path(path).name

    def _get_document_type(self, from_upload: bool) -> str:
        """Get document type based on from_upload flag."""
        return "user_upload" if from_upload else "agent_output"

    def write_text(self, path: str, content: str, user_visible: bool = True) -> None:
        """Write text content to S3 via NestJS API."""
        self.write_bytes(path, content.encode("utf-8"), user_visible=user_visible)

    def read_text(self, path: str, from_upload: bool = False) -> str:
        """Read text content from S3."""
        return self.read_bytes(path, from_upload=from_upload).decode("utf-8")

    def exists(self, path: str, from_upload: bool = False) -> bool:
        """Check if a document exists with matching type."""
        try:
            document_name = self._get_document_name(path)
            self._load_document_cache()

            existing_doc = self._document_cache.get(document_name)
            if not existing_doc:
                return False

            expected_type = self._get_document_type(from_upload)
            return bool(existing_doc["documentType"] == expected_type)
        except Exception:
            return False

    def write_bytes(self, path: str, content: bytes, user_visible: bool = True) -> None:
        """Write binary content to S3 via NestJS API."""
        client = self._get_client()
        document_name = self._get_document_name(path)
        file_name = document_name

        self._load_document_cache()
        existing_doc = self._document_cache.get(document_name)

        if existing_doc:
            # New version of existing document
            document_id = existing_doc["id"]

            upload_response = client.request_upload_url_sync(
                document_id=document_id,
                file_name=file_name,
                file_size=len(content),
                comment=f"Updated by {self.work_dir} agent",
            )

            client.upload_to_s3_sync(
                presigned_url=upload_response["presignedUrl"],
                content=content,
            )

            complete_response = client.complete_version_sync(
                document_id=document_id,
                upload_id=upload_response["uploadId"],
            )

            storage_path = complete_response.get("version", {}).get(
                "storagePath", existing_doc["storagePath"]
            )
            self._update_cache_entry(
                document_name=document_name,
                document_id=document_id,
                document_type=existing_doc["documentType"],
                storage_path=storage_path,
            )
        else:
            # New document
            document_type = self._get_document_type(from_upload=False)

            create_response = client.create_document_sync(
                thread_id=self.thread_id,
                document_name=document_name,
                file_name=file_name,
                file_size=len(content),
                document_type=document_type,
                comment=f"Created by {self.work_dir} agent",
                user_visible=user_visible,
            )

            client.upload_to_s3_sync(
                presigned_url=create_response["presignedUrl"],
                content=content,
            )

            client.complete_version_sync(
                document_id=create_response["documentId"],
                upload_id=create_response["uploadId"],
            )

            self._update_cache_entry(
                document_name=document_name,
                document_id=create_response["documentId"],
                document_type=document_type,
                storage_path=create_response["s3Key"],
            )

    def read_bytes(self, path: str, from_upload: bool = False) -> bytes:
        """Read binary content from S3."""
        client = self._get_client()
        document_name = self._get_document_name(path)

        self._load_document_cache()

        existing_doc = self._document_cache.get(document_name)
        if not existing_doc:
            raise FileNotFoundError(f"Document not found: {document_name}")

        expected_type = self._get_document_type(from_upload)
        if existing_doc["documentType"] != expected_type:
            raise FileNotFoundError(
                f"Document {document_name} has type {existing_doc['documentType']}, "
                f"expected {expected_type}"
            )

        document_id = existing_doc["id"]
        download_response = client.get_download_url_sync(document_id)

        return client.download_from_s3_sync(download_response["presignedUrl"])

    def get_path(self, *parts: str) -> str:
        """Join path parts (S3 uses forward slashes)."""
        return "/".join(parts)

    def cleanup(self) -> None:
        """Close the HTTP client session."""
        if self._client:
            self._client.close_sync()


def create_storage_backend(
    work_dir: str,
    nestjs_token: str | SecretStr | None = None,
    thread_id: str | None = None,
    nestjs_api_url: str | None = None,
) -> StorageBackend:
    """
    Factory function to create the appropriate storage backend.

    Uses S3 backend if USE_S3=true environment variable is set,
    otherwise uses local filesystem.

    Args:
        work_dir: Working directory name
        nestjs_token: NestJS JWT token (required for S3)
        thread_id: Thread ID for multi-tenant isolation
        nestjs_api_url: NestJS API base URL (defaults to NEST_URL env var)

    Returns:
        StorageBackend instance (S3 or Local)

    Raises:
        ValueError: If S3 is enabled but required parameters are missing

    Example:
        # Local storage
        backend = create_storage_backend("output")

        # S3 storage (with USE_S3=true)
        backend = create_storage_backend(
            "output",
            nestjs_token=token,
            thread_id="thread_123",
        )
    """
    if os.getenv("USE_S3", "").lower() == "true":
        api_url = nestjs_api_url or os.getenv("NEST_URL")
        if not api_url:
            raise ValueError("NEST_URL must be set when using S3 storage")
        if not nestjs_token:
            raise ValueError("NestJS token is required for S3 storage")
        if not thread_id:
            raise ValueError("Thread ID is required for S3 storage")

        logger.info("Using S3 storage backend (via NestJS)")
        token = nestjs_token if isinstance(nestjs_token, SecretStr) else SecretStr(nestjs_token)
        return S3StorageBackend(api_url, work_dir, token, thread_id)
    else:
        logger.info("Using local storage backend")
        return LocalStorageBackend(work_dir, thread_id)


@contextmanager
def storage_context(
    work_dir: str,
    nestjs_token: str | SecretStr | None = None,
    thread_id: str | None = None,
    nestjs_api_url: str | None = None,
) -> Any:
    """
    Context manager for storage backend with automatic cleanup.

    Creates the appropriate storage backend based on environment configuration
    and ensures cleanup on exit.

    Args:
        work_dir: Working directory name
        nestjs_token: NestJS JWT token (required for S3)
        thread_id: Thread ID for multi-tenant isolation
        nestjs_api_url: NestJS API base URL (defaults to NEST_URL env var)

    Yields:
        StorageBackend instance

    Example:
        with storage_context("output", nestjs_token=token, thread_id=tid) as storage:
            storage.write_text("result.json", json.dumps(data))
            content = storage.read_text("input.txt", from_upload=True)
        # Automatic cleanup on exit
    """
    backend = create_storage_backend(
        work_dir,
        nestjs_token=nestjs_token,
        thread_id=thread_id,
        nestjs_api_url=nestjs_api_url,
    )
    try:
        yield backend
    finally:
        backend.cleanup()


@asynccontextmanager
async def directory_context(
    file_access_params: FileAccessExtensionParams,
    keep: bool = False,
    local_dir: str | None = None,
) -> Any:
    """
    Async context manager that yields a local directory with S3 files downloaded.

    Downloads all user_upload documents from S3 to a temporary directory.
    Useful for agents that need to work with files locally (e.g., PDF processing).

    Args:
        file_access_params: File access extension parameters from request metadata
        keep: If True, don't delete temp directory on exit (for debugging)
        local_dir: If set and USE_S3 is false, use this directory instead of downloading

    Yields:
        Path to temporary directory containing downloaded files

    Example:
        params = FileAccessExtensionParams.model_validate(
            metadata[FILE_ACCESS_EXTENSION_URI_V2]
        )
        async with directory_context(params) as tmp_dir:
            files = os.listdir(tmp_dir)
            for f in files:
                content = Path(tmp_dir, f).read_bytes()
        # Temp directory auto-cleaned
    """
    use_s3 = os.getenv("USE_S3", "").lower() == "true"

    if use_s3:
        async with _s3_temp_directory_context(
            token=file_access_params.access_token,
            thread_id=file_access_params.context.thread_id,
            keep=keep,
        ) as tmp_dir:
            yield tmp_dir
    else:
        # Use local directory for development
        local_path: str = local_dir or os.getenv("LOCAL_PDF_DIR", "./input") or "./input"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local directory not found: {local_path}")
        yield local_path


@asynccontextmanager
async def _s3_temp_directory_context(
    token: SecretStr,
    thread_id: str,
    keep: bool = False,
) -> Any:
    """
    Internal context manager for downloading S3 files to temp directory.

    Downloads all user_upload documents from S3 via NestJS API into a
    unique temp namespace folder. Auto-cleans on exit unless keep=True.

    Args:
        token: JWT authentication token for NestJS API
        thread_id: Thread ID to download documents from
        keep: If True, don't delete temp directory on exit
    """
    if not thread_id:
        raise ValueError("Thread ID is required to access files")

    nestjs_api_url = os.getenv("NEST_URL")
    if not nestjs_api_url:
        raise ValueError("NEST_URL must be set when using S3 storage")

    client = NestJSClient(nestjs_api_url, token)

    # Generate safe namespace name
    safe_id = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    prefix_name = f"tmp_{safe_id}_"
    namespace_dir = tempfile.mkdtemp(prefix=prefix_name)

    try:
        logger.info(f"Listing documents in thread {thread_id}...")
        documents = client.list_documents_sync(thread_id)

        # Filter for user_upload documents only
        user_uploads = [doc for doc in documents if doc.get("documentType") == "user_upload"]

        if not user_uploads:
            logger.warning(f"No user_upload documents found in thread {thread_id}")

        logger.info(f"Downloading {len(user_uploads)} documents from S3...")

        for idx, doc in enumerate(user_uploads, 1):
            document_id = doc["id"]
            document_name = doc["name"]

            download_response = client.get_download_url_sync(document_id)
            content = client.download_from_s3_sync(download_response["presignedUrl"])

            local_path = os.path.join(namespace_dir, document_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as out:
                out.write(content)

            logger.debug(f"Downloaded {idx}/{len(user_uploads)}: {document_name}")

        if user_uploads:
            logger.info(f"Completed downloading {len(user_uploads)} documents from S3")

        yield namespace_dir

    finally:
        client.close_sync()

        if not keep:
            shutil.rmtree(namespace_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {namespace_dir}")
