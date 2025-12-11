# Storage API Reference

## StorageBackend

::: health_universe_a2a.StorageBackend
    options:
      show_root_heading: true
      members:
        - write_text
        - read_text
        - write_bytes
        - read_bytes
        - exists
        - get_path
        - cleanup

## LocalStorageBackend

::: health_universe_a2a.LocalStorageBackend
    options:
      show_root_heading: true

## S3StorageBackend

::: health_universe_a2a.S3StorageBackend
    options:
      show_root_heading: true

## NestJSClient

::: health_universe_a2a.NestJSClient
    options:
      show_root_heading: true
      members:
        - list_documents
        - list_documents_sync
        - get_download_url
        - get_download_url_sync
        - download_from_s3
        - download_from_s3_sync
        - create_document
        - create_document_sync
        - upload_to_s3
        - upload_to_s3_sync

## Functions

### create_storage_backend

::: health_universe_a2a.create_storage_backend

### storage_context

::: health_universe_a2a.storage_context

### directory_context

::: health_universe_a2a.directory_context
