"""
HTTP client for interacting with the NestJS document storage API.

Handles document creation, upload URLs, version completion, and downloads
for S3 storage via the Health Universe platform.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class NestJSClient:
    """
    Async HTTP client for NestJS document storage API.

    Provides methods for:
    - Creating new documents with presigned upload URLs
    - Requesting upload URLs for new versions
    - Completing version uploads
    - Getting presigned download URLs
    - Listing documents in a thread
    - Uploading/downloading directly to/from S3

    Both async and sync versions of all methods are provided for
    compatibility with ThreadPoolExecutor contexts.

    Example:
        # Async usage
        client = NestJSClient(base_url="https://apps.healthuniverse.com/api/v1", token=token)
        try:
            docs = await client.list_documents(thread_id)
            for doc in docs:
                url_info = await client.get_download_url(doc["id"])
                content = await client.download_from_s3(url_info["presignedUrl"])
        finally:
            await client.close()

        # Sync usage (for ThreadPoolExecutor)
        docs = client.list_documents_sync(thread_id)
        for doc in docs:
            url_info = client.get_download_url_sync(doc["id"])
            content = client.download_from_s3_sync(url_info["presignedUrl"])
        client.close_sync()
    """

    def __init__(self, base_url: str, token: SecretStr):
        """
        Initialize NestJS client.

        Args:
            base_url: Base URL of the NestJS API (e.g., "https://apps.healthuniverse.com/api/v1")
            token: JWT authentication token (SecretStr for secure handling)
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._debug = os.getenv("DEBUG_HTTP_REQUESTS", "false").lower() == "true"

    def _get_headers(self) -> dict[str, str]:
        """Get default headers with auth token."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token.get_secret_value()}",
        }

    @asynccontextmanager
    async def _get_async_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get or create async HTTP client with auth headers."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                headers=self._get_headers(),
                timeout=30.0,
            )

        try:
            yield self._async_client
        finally:
            pass  # Client cleanup happens in close()

    def _get_sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client with auth headers."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                headers=self._get_headers(),
                timeout=30.0,
            )
        return self._sync_client

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close_sync(self) -> None:
        """Close the synchronous HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        retries: int = 3,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Make async HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/documents")
            json: Request body (for POST/PUT)
            params: Query parameters
            retries: Number of retry attempts

        Returns:
            Response JSON as dict or list

        Raises:
            httpx.HTTPError: On HTTP errors after all retries
        """
        url = f"{self.base_url}{path}"

        if self._debug:
            logger.debug(f"{method} {url}")
            if json:
                logger.debug(f"Request body: {json}")
            if params:
                logger.debug(f"Query params: {params}")

        async with self._get_async_client() as client:
            for attempt in range(retries):
                try:
                    response = await client.request(
                        method,
                        url,
                        json=json,
                        params=params,
                    )

                    if self._debug:
                        logger.debug(f"Response status: {response.status_code}")
                        logger.debug(f"Response body: {response.text[:500]}")

                    response.raise_for_status()

                    if response.text:
                        return cast(
                            dict[str, Any] | list[dict[str, Any]], response.json()
                        )
                    return {}

                except httpx.HTTPStatusError as e:
                    # Don't retry on 4xx errors (client errors)
                    if 400 <= e.response.status_code < 500:
                        logger.error(f"Client error {e.response.status_code}: {e}")
                        raise

                    # Retry on 5xx errors (server errors)
                    if attempt < retries - 1:
                        logger.warning(
                            f"Server error {e.response.status_code}, "
                            f"retrying... (attempt {attempt + 1}/{retries})"
                        )
                        continue
                    raise

                except httpx.HTTPError as e:
                    if attempt < retries - 1:
                        logger.warning(
                            f"Request failed: {e}, retrying... (attempt {attempt + 1}/{retries})"
                        )
                        continue
                    raise

        raise RuntimeError(f"Failed after {retries} attempts")

    def _request_sync(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        retries: int = 3,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Make synchronous HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/documents")
            json: Request body (for POST/PUT)
            params: Query parameters
            retries: Number of retry attempts

        Returns:
            Response JSON as dict or list

        Raises:
            httpx.HTTPError: On HTTP errors after all retries
        """
        url = f"{self.base_url}{path}"

        if self._debug:
            logger.debug(f"{method} {url}")
            if json:
                logger.debug(f"Request body: {json}")
            if params:
                logger.debug(f"Query params: {params}")

        client = self._get_sync_client()

        for attempt in range(retries):
            try:
                response = client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                )

                if self._debug:
                    logger.debug(f"Response status: {response.status_code}")
                    logger.debug(f"Response body: {response.text[:500]}")

                response.raise_for_status()

                if response.text:
                    return cast(
                        dict[str, Any] | list[dict[str, Any]], response.json()
                    )
                return {}

            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error {e.response.status_code}: {e}")
                    raise

                # Retry on 5xx errors (server errors)
                if attempt < retries - 1:
                    logger.warning(
                        f"Server error {e.response.status_code}, "
                        f"retrying... (attempt {attempt + 1}/{retries})"
                    )
                    continue
                raise

            except httpx.HTTPError as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Request failed: {e}, retrying... (attempt {attempt + 1}/{retries})"
                    )
                    continue
                raise

        raise RuntimeError(f"Failed after {retries} attempts")

    # ========== Sync API Methods ==========

    async def list_documents(self, thread_id: str) -> list[dict[str, Any]]:
        """
        List all documents in a thread.

        Args:
            thread_id: Thread ID to list documents for

        Returns:
            List of document metadata dicts with keys:
            - id: Document UUID
            - name: Document name
            - documentType: 'user_upload' or 'agent_output'
            - storagePath: S3 storage path
            - latestVersion: Latest version metadata
        """
        result = await self._request(
            "POST",
            "/documents/search",
            json={"threadId": thread_id},
        )
        return result if isinstance(result, list) else []

    async def create_document(
        self,
        thread_id: str,
        document_name: str,
        file_name: str,
        file_size: int | None = None,
        content_hash: str | None = None,
        document_type: str = "agent_output",
        comment: str | None = None,
        user_visible: bool = True,
    ) -> dict[str, Any]:
        """
        Create a new document and get presigned upload URL.

        Args:
            thread_id: Thread ID
            document_name: Document name (display name)
            file_name: File name for storage
            file_size: File size in bytes (optional)
            content_hash: Content hash for verification (optional)
            document_type: 'user_upload' or 'agent_output'
            comment: Optional comment for version
            user_visible: Whether document is visible to users (default: True)

        Returns:
            Dict with keys:
            - documentId: Document UUID
            - uploadId: Upload UUID
            - presignedUrl: Presigned PUT URL
            - s3Key: S3 storage key
            - versionNumber: Version number (always 1)
            - expiresIn: Seconds until URL expires (600)
            - expiresAt: ISO timestamp of expiration
        """
        payload: dict[str, Any] = {
            "threadId": thread_id,
            "documentName": document_name,
            "fileName": file_name,
            "documentType": document_type,
            "userVisible": user_visible,
        }

        if file_size is not None:
            payload["fileSize"] = file_size
        if content_hash is not None:
            payload["contentHash"] = content_hash
        if comment is not None:
            payload["comment"] = comment

        result = await self._request("POST", "/documents", json=payload)
        return result if isinstance(result, dict) else {}

    async def request_upload_url(
        self,
        document_id: str,
        file_name: str,
        file_size: int | None = None,
        content_hash: str | None = None,
        base_version_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Request presigned upload URL for a new version.

        Args:
            document_id: Document UUID
            file_name: File name for storage
            file_size: File size in bytes (optional)
            content_hash: Content hash for verification (optional)
            base_version_id: Base version UUID for optimistic locking (optional)
            comment: Optional comment for version

        Returns:
            Dict with keys:
            - uploadId: Upload UUID
            - presignedUrl: Presigned PUT URL
            - versionNumber: Version number
            - expiresIn: Seconds until URL expires (300)
        """
        payload: dict[str, Any] = {"fileName": file_name}

        if file_size is not None:
            payload["fileSize"] = file_size
        if content_hash is not None:
            payload["contentHash"] = content_hash
        if base_version_id is not None:
            payload["baseVersionId"] = base_version_id
        if comment is not None:
            payload["comment"] = comment

        result = await self._request(
            "POST",
            f"/documents/{document_id}/upload-url",
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    async def complete_version(
        self,
        document_id: str,
        upload_id: str,
    ) -> dict[str, Any]:
        """
        Complete version upload after S3 PUT succeeds.

        Args:
            document_id: Document UUID
            upload_id: Upload UUID from request_upload_url

        Returns:
            Dict with keys:
            - success: True
            - version: Version metadata dict
        """
        payload = {"uploadId": upload_id}

        result = await self._request(
            "POST",
            f"/documents/{document_id}/complete-version",
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    async def get_download_url(self, document_id: str) -> dict[str, Any]:
        """
        Get presigned download URL for current version.

        Args:
            document_id: Document UUID

        Returns:
            Dict with keys:
            - presignedUrl: Presigned GET URL
            - fileName: Document file name
            - versionId: Version UUID
            - expiresIn: Seconds until URL expires (180)
        """
        result = await self._request(
            "GET",
            f"/documents/{document_id}/download-url",
        )
        return result if isinstance(result, dict) else {}

    async def upload_to_s3(
        self,
        presigned_url: str,
        content: bytes,
    ) -> None:
        """
        Upload content directly to S3 using presigned URL.

        Args:
            presigned_url: Presigned PUT URL from create_document or request_upload_url
            content: File content as bytes

        Raises:
            httpx.HTTPError: On upload failure
        """
        if self._debug:
            logger.debug(f"Uploading {len(content)} bytes to S3")

        # Use a separate client without auth headers for S3
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.put(
                presigned_url,
                content=content,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()

            if self._debug:
                logger.debug(f"S3 upload complete: {response.status_code}")

    async def download_from_s3(self, presigned_url: str) -> bytes:
        """
        Download content from S3 using presigned URL.

        Args:
            presigned_url: Presigned GET URL from get_download_url

        Returns:
            File content as bytes

        Raises:
            httpx.HTTPError: On download failure
        """
        if self._debug:
            logger.debug("Downloading from S3")

        # Use a separate client without auth headers for S3
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(presigned_url)
            response.raise_for_status()
            content = response.content

            if self._debug:
                logger.debug(f"S3 download complete: {len(content)} bytes")

            return content

    # ========== Sync API Methods ==========

    def list_documents_sync(self, thread_id: str) -> list[dict[str, Any]]:
        """Synchronous version of list_documents."""
        result = self._request_sync(
            "POST",
            "/documents/search",
            json={"threadId": thread_id},
        )
        return result if isinstance(result, list) else []

    def create_document_sync(
        self,
        thread_id: str,
        document_name: str,
        file_name: str,
        file_size: int | None = None,
        content_hash: str | None = None,
        document_type: str = "agent_output",
        comment: str | None = None,
        user_visible: bool = True,
    ) -> dict[str, Any]:
        """Synchronous version of create_document."""
        payload: dict[str, Any] = {
            "threadId": thread_id,
            "documentName": document_name,
            "fileName": file_name,
            "documentType": document_type,
            "userVisible": user_visible,
        }

        if file_size is not None:
            payload["fileSize"] = file_size
        if content_hash is not None:
            payload["contentHash"] = content_hash
        if comment is not None:
            payload["comment"] = comment

        result = self._request_sync("POST", "/documents", json=payload)
        return result if isinstance(result, dict) else {}

    def request_upload_url_sync(
        self,
        document_id: str,
        file_name: str,
        file_size: int | None = None,
        content_hash: str | None = None,
        base_version_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous version of request_upload_url."""
        payload: dict[str, Any] = {"fileName": file_name}

        if file_size is not None:
            payload["fileSize"] = file_size
        if content_hash is not None:
            payload["contentHash"] = content_hash
        if base_version_id is not None:
            payload["baseVersionId"] = base_version_id
        if comment is not None:
            payload["comment"] = comment

        result = self._request_sync(
            "POST",
            f"/documents/{document_id}/upload-url",
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    def complete_version_sync(
        self,
        document_id: str,
        upload_id: str,
    ) -> dict[str, Any]:
        """Synchronous version of complete_version."""
        payload = {"uploadId": upload_id}

        result = self._request_sync(
            "POST",
            f"/documents/{document_id}/complete-version",
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    def get_download_url_sync(self, document_id: str) -> dict[str, Any]:
        """Synchronous version of get_download_url."""
        result = self._request_sync(
            "GET",
            f"/documents/{document_id}/download-url",
        )
        return result if isinstance(result, dict) else {}

    def upload_to_s3_sync(
        self,
        presigned_url: str,
        content: bytes,
    ) -> None:
        """
        Synchronous upload to S3 using presigned URL.

        Args:
            presigned_url: Presigned PUT URL
            content: File content as bytes

        Raises:
            httpx.HTTPError: On upload failure
        """
        if self._debug:
            logger.debug(f"Uploading {len(content)} bytes to S3")

        # Use a separate client without auth headers for S3
        with httpx.Client(timeout=300.0) as client:
            response = client.put(
                presigned_url,
                content=content,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()

            if self._debug:
                logger.debug(f"S3 upload complete: {response.status_code}")

    def download_from_s3_sync(self, presigned_url: str) -> bytes:
        """
        Synchronous download from S3 using presigned URL.

        Args:
            presigned_url: Presigned GET URL

        Returns:
            File content as bytes

        Raises:
            httpx.HTTPError: On download failure
        """
        if self._debug:
            logger.debug("Downloading from S3")

        # Use a separate client without auth headers for S3
        with httpx.Client(timeout=300.0) as client:
            response = client.get(presigned_url)
            response.raise_for_status()
            content = response.content

            if self._debug:
                logger.debug(f"S3 download complete: {len(content)} bytes")

            return content
