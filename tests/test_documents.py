"""Tests for DocumentClient and Document classes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from health_universe_a2a.documents import Document, DocumentClient


class TestDocument:
    """Tests for Document dataclass."""

    def test_from_dict_minimal(self):
        """Test creating Document from minimal dict."""
        data = {
            "id": "doc-123",
            "name": "Test Document",
            "fileName": "test.pdf",
            "documentType": "user_upload",
        }
        doc = Document.from_dict(data)

        assert doc.id == "doc-123"
        assert doc.name == "Test Document"
        assert doc.filename == "test.pdf"
        assert doc.document_type == "user_upload"
        assert doc.storage_path is None
        assert doc.latest_version is None
        assert doc.latest_version_id is None
        assert doc.user_visible is True

    def test_from_dict_full(self):
        """Test creating Document from full dict with version info."""
        data = {
            "id": "doc-456",
            "name": "Full Document",
            "fileName": "full.docx",
            "documentType": "agent_output",
            "storagePath": "s3://bucket/path/full.docx",
            "userVisible": False,
            "latestVersion": {
                "id": "version-789",
                "versionNumber": 3,
            },
        }
        doc = Document.from_dict(data)

        assert doc.id == "doc-456"
        assert doc.name == "Full Document"
        assert doc.filename == "full.docx"
        assert doc.document_type == "agent_output"
        assert doc.storage_path == "s3://bucket/path/full.docx"
        assert doc.latest_version == 3
        assert doc.latest_version_id == "version-789"
        assert doc.user_visible is False

    def test_from_dict_alternative_name_field(self):
        """Test Document handles documentName field."""
        data = {
            "id": "doc-alt",
            "documentName": "Alt Name Document",
            "fileName": "alt.txt",
            "documentType": "user_upload",
        }
        doc = Document.from_dict(data)
        assert doc.name == "Alt Name Document"


class TestDocumentClient:
    """Tests for DocumentClient class."""

    @pytest.fixture
    def mock_nest_client(self):
        """Create a mock NestJSClient."""
        with patch("health_universe_a2a.documents.NestJSClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def client(self, mock_nest_client):
        """Create DocumentClient with mocked NestJSClient."""
        return DocumentClient(
            base_url="https://api.example.com",
            access_token="test-token",
            thread_id="thread-123",
        )

    @pytest.mark.asyncio
    async def test_list_documents(self, client, mock_nest_client):
        """Test listing documents."""
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "name": "Doc 1",
                    "fileName": "doc1.pdf",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
                {
                    "id": "doc-2",
                    "name": "Doc 2",
                    "fileName": "doc2.pdf",
                    "documentType": "agent_output",
                    "userVisible": False,
                },
            ]
        )

        docs = await client.list_documents()

        assert len(docs) == 1  # Only visible docs
        assert docs[0].id == "doc-1"
        mock_nest_client.list_documents.assert_called_once_with("thread-123")

    @pytest.mark.asyncio
    async def test_list_documents_include_hidden(self, client, mock_nest_client):
        """Test listing documents including hidden ones."""
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "name": "Doc 1",
                    "fileName": "doc1.pdf",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
                {
                    "id": "doc-2",
                    "name": "Doc 2",
                    "fileName": "doc2.pdf",
                    "documentType": "agent_output",
                    "userVisible": False,
                },
            ]
        )

        docs = await client.list_documents(include_hidden=True)

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_name(self, client, mock_nest_client):
        """Test filtering documents by name."""
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "name": "Protocol Document",
                    "fileName": "protocol.pdf",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
                {
                    "id": "doc-2",
                    "name": "Report",
                    "fileName": "report.pdf",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
            ]
        )

        matches = await client.filter_by_name("protocol")

        assert len(matches) == 1
        assert matches[0].name == "Protocol Document"

    @pytest.mark.asyncio
    async def test_filter_by_name_matches_filename(self, client, mock_nest_client):
        """Test filter_by_name also matches filename."""
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "name": "Data File",
                    "fileName": "data.csv",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
            ]
        )

        matches = await client.filter_by_name(".csv")

        assert len(matches) == 1

    @pytest.mark.asyncio
    async def test_get_document(self, client, mock_nest_client):
        """Test getting a document by ID."""
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "doc-target",
                    "name": "Target Doc",
                    "fileName": "target.pdf",
                    "documentType": "user_upload",
                    "userVisible": True,
                },
            ]
        )

        doc = await client.get_document("doc-target")

        assert doc.id == "doc-target"
        assert doc.name == "Target Doc"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client, mock_nest_client):
        """Test getting a non-existent document raises ValueError."""
        mock_nest_client.list_documents = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="Document not found"):
            await client.get_document("non-existent")

    @pytest.mark.asyncio
    async def test_download(self, client, mock_nest_client):
        """Test downloading document content."""
        mock_nest_client.get_download_url = AsyncMock(
            return_value={"presignedUrl": "https://s3.example.com/doc"}
        )
        mock_nest_client.download_from_s3 = AsyncMock(return_value=b"file content")

        content = await client.download("doc-123")

        assert content == b"file content"
        mock_nest_client.get_download_url.assert_called_once_with("doc-123")

    @pytest.mark.asyncio
    async def test_download_no_url(self, client, mock_nest_client):
        """Test download raises error when no presigned URL."""
        mock_nest_client.get_download_url = AsyncMock(return_value={})

        with pytest.raises(ValueError, match="Failed to get download URL"):
            await client.download("doc-123")

    @pytest.mark.asyncio
    async def test_download_text(self, client, mock_nest_client):
        """Test downloading document as text."""
        mock_nest_client.get_download_url = AsyncMock(
            return_value={"presignedUrl": "https://s3.example.com/doc"}
        )
        mock_nest_client.download_from_s3 = AsyncMock(return_value=b"text content")

        content = await client.download_text("doc-123")

        assert content == "text content"

    @pytest.mark.asyncio
    async def test_write_string_content(self, client, mock_nest_client):
        """Test writing string content creates document."""
        mock_nest_client.create_document = AsyncMock(
            return_value={
                "documentId": "new-doc-id",
                "uploadId": "upload-123",
                "presignedUrl": "https://s3.example.com/upload",
                "s3Key": "path/to/file",
                "versionNumber": 1,
            }
        )
        mock_nest_client.upload_to_s3 = AsyncMock()
        mock_nest_client.complete_version = AsyncMock()

        doc = await client.write(
            name="Test Output",
            content="Hello, world!",
            filename="test.txt",
        )

        assert doc.id == "new-doc-id"
        assert doc.name == "Test Output"
        assert doc.filename == "test.txt"
        mock_nest_client.create_document.assert_called_once()
        mock_nest_client.upload_to_s3.assert_called_once()
        mock_nest_client.complete_version.assert_called_once_with("new-doc-id", "upload-123")

    @pytest.mark.asyncio
    async def test_write_bytes_content(self, client, mock_nest_client):
        """Test writing bytes content."""
        mock_nest_client.create_document = AsyncMock(
            return_value={
                "documentId": "new-doc-id",
                "uploadId": "upload-123",
                "presignedUrl": "https://s3.example.com/upload",
            }
        )
        mock_nest_client.upload_to_s3 = AsyncMock()
        mock_nest_client.complete_version = AsyncMock()

        doc = await client.write(
            name="Binary File",
            content=b"\x00\x01\x02",
            filename="data.bin",
        )

        assert doc.id == "new-doc-id"

    @pytest.mark.asyncio
    async def test_write_auto_filename_json(self, client, mock_nest_client):
        """Test auto-generating filename for JSON content."""
        mock_nest_client.create_document = AsyncMock(
            return_value={
                "documentId": "doc-id",
                "uploadId": "upload-id",
                "presignedUrl": "https://s3.example.com/upload",
            }
        )
        mock_nest_client.upload_to_s3 = AsyncMock()
        mock_nest_client.complete_version = AsyncMock()

        doc = await client.write(
            name="JSON Data",
            content='{"key": "value"}',
        )

        assert doc.filename == "json_data.json"

    @pytest.mark.asyncio
    async def test_write_auto_filename_markdown(self, client, mock_nest_client):
        """Test auto-generating filename for Markdown content."""
        mock_nest_client.create_document = AsyncMock(
            return_value={
                "documentId": "doc-id",
                "uploadId": "upload-id",
                "presignedUrl": "https://s3.example.com/upload",
            }
        )
        mock_nest_client.upload_to_s3 = AsyncMock()
        mock_nest_client.complete_version = AsyncMock()

        doc = await client.write(
            name="My Report",
            content="# Heading\n\nContent here",
        )

        assert doc.filename == "my_report.md"

    @pytest.mark.asyncio
    async def test_write_failure(self, client, mock_nest_client):
        """Test write raises error on creation failure."""
        mock_nest_client.create_document = AsyncMock(return_value={})

        with pytest.raises(ValueError, match="Failed to create document"):
            await client.write(name="Test", content="content")

    @pytest.mark.asyncio
    async def test_update_document(self, client, mock_nest_client):
        """Test updating an existing document."""
        # Mock get_document
        mock_nest_client.list_documents = AsyncMock(
            return_value=[
                {
                    "id": "existing-doc",
                    "name": "Existing",
                    "fileName": "existing.txt",
                    "documentType": "agent_output",
                    "userVisible": True,
                }
            ]
        )
        mock_nest_client.request_upload_url = AsyncMock(
            return_value={
                "uploadId": "upload-456",
                "presignedUrl": "https://s3.example.com/upload",
                "versionNumber": 2,
            }
        )
        mock_nest_client.upload_to_s3 = AsyncMock()
        mock_nest_client.complete_version = AsyncMock()

        doc = await client.update(
            document_id="existing-doc",
            content="Updated content",
            comment="Version 2",
        )

        assert doc.id == "existing-doc"
        assert doc.latest_version == 2

    @pytest.mark.asyncio
    async def test_close(self, client, mock_nest_client):
        """Test closing the client."""
        mock_nest_client.close = AsyncMock()

        await client.close()

        mock_nest_client.close.assert_called_once()

    def test_slugify(self, client):
        """Test filename slugification."""
        assert client._slugify("Hello World") == "hello_world"
        assert client._slugify("Test!@#$%File") == "testfile"
        assert client._slugify("Multiple   Spaces") == "multiple_spaces"
        assert client._slugify("") == "document"
        assert client._slugify("___leading___") == "leading"
