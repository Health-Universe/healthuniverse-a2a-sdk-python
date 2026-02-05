"""Document operations for Health Universe agents.

The DocumentClient provides a simple interface for document operations in agent threads.
It wraps the NestJS document API with convenience methods for common operations.

Typical usage (via context.document_client):
    async def process_message(self, message: str, context: AgentContext) -> str:
        # List all documents in the thread
        docs = await context.document_client.list_documents()

        # Read a document's content
        content = await context.document_client.download_text(docs[0].id)

        # Write a new document (convenience method handles 3-step process)
        await context.document_client.write(
            "Analysis Results",
            json.dumps(results),
            filename="results.json"
        )

        # Filter documents by name
        protocols = await context.document_client.filter_by_name("protocol")

        return "Done!"
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from health_universe_a2a.nest_client import NestJSClient

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Metadata for a document in the thread.

    Attributes:
        id: Document UUID
        name: Display name (e.g., "Clinical Protocol")
        filename: Storage filename (e.g., "protocol.pdf")
        document_type: "user_upload" or "agent_output"
        storage_path: S3 storage path
        latest_version: Version number of latest version
        latest_version_id: UUID of latest version
        user_visible: Whether document is visible to users

    Example:
        docs = await context.document_client.list_documents()
        for doc in docs:
            print(f"{doc.name} (v{doc.latest_version})")
    """

    id: str
    name: str
    filename: str
    document_type: str
    storage_path: str | None
    latest_version: int | None
    latest_version_id: str | None
    user_visible: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from API response dict."""
        latest_version = data.get("latestVersion", {})
        return cls(
            id=data["id"],
            name=data.get("name", data.get("documentName", "")),
            filename=data.get("fileName", ""),
            document_type=data.get("documentType", ""),
            storage_path=data.get("storagePath"),
            latest_version=latest_version.get("versionNumber") if latest_version else None,
            latest_version_id=latest_version.get("id") if latest_version else None,
            user_visible=data.get("userVisible", True),
        )


class DocumentClient:
    """
    Client for document operations in a Health Universe thread.

    Provides a simple interface for listing, reading, writing, and searching
    documents. The `write()` method handles the 3-step upload process
    (create → S3 upload → complete) automatically.

    This client is typically accessed via `context.document_client` in agent methods.

    Document Content Tiers:
        The platform stores multiple representations of uploaded documents:

        1. **Raw** - Original uploaded file (PDF, DOCX, images, etc.) or agent output.
           Access via `download()` (bytes) or `download_text()` (text files only).

        2. **Extracted** (future) - Text extracted from raw uploads, converted to
           markdown by the platform. Will be accessible via `download_extracted()`.

        3. **Semantic Search** (future) - Vector embeddings generated from extracted
           text. Will enable `semantic_search(query)` across thread documents.

    Example:
        async def process_message(self, message: str, context: AgentContext) -> str:
            # List documents
            docs = await context.document_client.list_documents()
            print(f"Found {len(docs)} documents")

            # Find and read a protocol document
            protocols = await context.document_client.filter_by_name("protocol")
            if protocols:
                content = await context.document_client.download_text(protocols[0].id)
                print(f"Protocol content: {content[:100]}...")

            # Write analysis results
            await context.document_client.write(
                "Analysis Results",
                json.dumps({"score": 0.95}),
                filename="results.json"
            )

            return "Analysis complete!"

    Attributes:
        thread_id: The thread ID for document operations
    """

    def __init__(self, base_url: str, access_token: str, thread_id: str):
        """
        Initialize DocumentClient.

        Args:
            base_url: NestJS API base URL (e.g., "https://apps.healthuniverse.com/api/v1")
            access_token: JWT access token for authentication
            thread_id: Thread ID for document operations
        """
        self.thread_id = thread_id
        self._client = NestJSClient(
            base_url=base_url,
            token=SecretStr(access_token),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

    # ========== Document Listing ==========

    async def list_documents(self, include_hidden: bool = False) -> list[Document]:
        """
        List all documents in the thread.

        Args:
            include_hidden: Include documents not visible to users (default: False)

        Returns:
            List of Document objects with metadata

        Example:
            docs = await context.document_client.list_documents()
            for doc in docs:
                print(f"- {doc.name} ({doc.document_type}, v{doc.latest_version})")
        """
        raw_docs = await self._client.list_documents(self.thread_id)
        docs = [Document.from_dict(d) for d in raw_docs]

        if not include_hidden:
            docs = [d for d in docs if d.user_visible]

        return docs

    async def filter_by_name(self, query: str) -> list[Document]:
        """
        Filter documents by name or filename (case-insensitive substring match).

        Args:
            query: Substring to match in document name or filename

        Returns:
            List of matching Document objects

        Example:
            # Find protocol documents
            protocols = await context.document_client.filter_by_name("protocol")

            # Find CSV files
            csvs = await context.document_client.filter_by_name(".csv")
        """
        docs = await self.list_documents(include_hidden=True)

        query_lower = query.lower()
        matches = [
            d for d in docs if query_lower in d.name.lower() or query_lower in d.filename.lower()
        ]

        return matches

    # ========== Document Reading ==========

    async def get_document(self, document_id: str) -> Document:
        """
        Get document metadata by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document object with metadata

        Raises:
            ValueError: If document not found

        Example:
            doc = await context.document_client.get_document(doc_id)
            print(f"Document: {doc.name} (v{doc.latest_version})")
        """
        docs = await self.list_documents(include_hidden=True)
        for doc in docs:
            if doc.id == document_id:
                return doc
        raise ValueError(f"Document not found: {document_id}")

    async def download(self, document_id: str) -> bytes:
        """
        Download document content as bytes.

        Args:
            document_id: Document UUID

        Returns:
            Document content as bytes

        Example:
            # Download binary file
            pdf_bytes = await context.document_client.download(doc_id)
            with open("output.pdf", "wb") as f:
                f.write(pdf_bytes)
        """
        url_info = await self._client.get_download_url(document_id)
        presigned_url = url_info.get("presignedUrl")
        if not presigned_url:
            raise ValueError(f"Failed to get download URL for document {document_id}")

        return await self._client.download_from_s3(presigned_url)

    async def download_text(self, document_id: str, encoding: str = "utf-8") -> str:
        """
        Download raw document content decoded as text.

        Note: This method downloads the raw file and decodes it as text.
        Only use for files that are natively text-based (CSV, JSON, TXT, MD).
        For PDFs, DOCX, and other binary formats, use download() to get raw bytes.

        Future: For extracted text from PDFs/DOCX (converted to markdown by the
        platform), a separate download_extracted() method will be added.

        Args:
            document_id: Document UUID
            encoding: Text encoding (default: "utf-8")

        Returns:
            Document content as string

        Example:
            # Read a CSV file
            csv_content = await context.document_client.download_text(csv_doc.id)
            lines = csv_content.split("\\n")

            # Read a JSON file
            json_content = await context.document_client.download_text(json_doc.id)
            data = json.loads(json_content)
        """
        content = await self.download(document_id)
        return content.decode(encoding)

    # ========== Document Writing ==========

    async def write(
        self,
        name: str,
        content: str | bytes,
        filename: str | None = None,
        document_type: str = "agent_output",
        user_visible: bool = True,
        comment: str | None = None,
    ) -> Document:
        """
        Write content to a new document.

        This is a convenience method that handles the 3-step upload process:
        1. Create document and get presigned upload URL
        2. Upload content to S3
        3. Complete the version

        Args:
            name: Display name for the document (e.g., "Analysis Results")
            content: File content as string or bytes
            filename: Storage filename (defaults to slugified name + extension)
            document_type: "agent_output" (default) or "user_upload"
            user_visible: Whether document appears in user's file list (default: True)
            comment: Optional version comment

        Returns:
            Document object for the created document

        Example:
            # Write JSON results
            await context.document_client.write(
                "Patient Analysis",
                json.dumps({"risk_score": 0.73}),
                filename="analysis.json"
            )

            # Write markdown report
            await context.document_client.write(
                "Clinical Summary",
                "# Summary\\n\\nPatient shows...",
                filename="summary.md"
            )

            # Write binary data
            await context.document_client.write(
                "Generated PDF",
                pdf_bytes,
                filename="report.pdf"
            )
        """
        # Convert string content to bytes
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        # Generate filename if not provided
        if not filename:
            filename = self._slugify(name)
            # Add extension based on content type detection
            if isinstance(content, str):
                if content.strip().startswith("{") or content.strip().startswith("["):
                    filename += ".json"
                elif content.strip().startswith("#") or content.strip().startswith("*"):
                    filename += ".md"
                else:
                    filename += ".txt"
            else:
                filename += ".bin"

        # Calculate content hash for integrity
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Step 1: Create document and get upload URL
        create_response = await self._client.create_document(
            thread_id=self.thread_id,
            document_name=name,
            file_name=filename,
            file_size=len(content_bytes),
            content_hash=content_hash,
            document_type=document_type,
            comment=comment,
            user_visible=user_visible,
        )

        document_id = create_response.get("documentId")
        upload_id = create_response.get("uploadId")
        presigned_url = create_response.get("presignedUrl")

        if not document_id or not upload_id or not presigned_url:
            raise ValueError(f"Failed to create document: {create_response}")

        # Step 2: Upload to S3
        await self._client.upload_to_s3(presigned_url, content_bytes)

        # Step 3: Complete the version
        await self._client.complete_version(document_id, upload_id)

        # Return Document object
        return Document(
            id=document_id,
            name=name,
            filename=filename,
            document_type=document_type,
            storage_path=create_response.get("s3Key"),
            latest_version=create_response.get("versionNumber", 1),
            latest_version_id=None,  # Not returned from create
            user_visible=user_visible,
        )

    async def update(
        self,
        document_id: str,
        content: str | bytes,
        comment: str | None = None,
        base_version_id: str | None = None,
    ) -> Document:
        """
        Update an existing document with new content (creates new version).

        Args:
            document_id: Document UUID to update
            content: New file content as string or bytes
            comment: Optional version comment
            base_version_id: Base version UUID for optimistic locking (optional)

        Returns:
            Updated Document object

        Example:
            # Update a document with new content
            updated = await context.document_client.update(
                doc.id,
                json.dumps(new_data),
                comment="Updated with new analysis results"
            )
            print(f"Updated to version {updated.latest_version}")
        """
        # Get current document metadata
        doc = await self.get_document(document_id)

        # Convert string content to bytes
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        # Calculate content hash
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Step 1: Request upload URL for new version
        upload_response = await self._client.request_upload_url(
            document_id=document_id,
            file_name=doc.filename,
            file_size=len(content_bytes),
            content_hash=content_hash,
            base_version_id=base_version_id,
            comment=comment,
        )

        upload_id = upload_response.get("uploadId")
        presigned_url = upload_response.get("presignedUrl")
        version_number = upload_response.get("versionNumber")

        if not upload_id or not presigned_url:
            raise ValueError(f"Failed to get upload URL: {upload_response}")

        # Step 2: Upload to S3
        await self._client.upload_to_s3(presigned_url, content_bytes)

        # Step 3: Complete the version
        await self._client.complete_version(document_id, upload_id)

        # Return updated Document
        return Document(
            id=document_id,
            name=doc.name,
            filename=doc.filename,
            document_type=doc.document_type,
            storage_path=doc.storage_path,
            latest_version=version_number,
            latest_version_id=None,  # Not returned from complete_version
            user_visible=doc.user_visible,
        )

    def _slugify(self, text: str) -> str:
        """Convert text to a safe filename slug."""
        # Convert to lowercase
        text = text.lower()
        # Replace spaces and special chars with underscores
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        # Remove leading/trailing underscores
        text = text.strip("_")
        return text or "document"
