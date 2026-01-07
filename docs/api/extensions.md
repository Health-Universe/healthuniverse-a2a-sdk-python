# Extensions API Reference

## Constants

### FILE_ACCESS_EXTENSION_URI

```python
FILE_ACCESS_EXTENSION_URI = "https://healthuniverse.com/ext/file_access/v2"
```

File access extension URI for document operations via NestJS/S3.

### BACKGROUND_JOB_EXTENSION_URI

```python
BACKGROUND_JOB_EXTENSION_URI = "https://healthuniverse.com/ext/background_job"
```

Background job extension for long-running tasks.

### HU_LOG_LEVEL_EXTENSION_URI

```python
HU_LOG_LEVEL_EXTENSION_URI = "https://healthuniverse.com/ext/log_level/v1"
```

Log level configuration extension.

## UpdateImportance

::: health_universe_a2a.UpdateImportance
    options:
      show_root_heading: true
      members:
        - ERROR
        - NOTICE
        - INFO
        - DEBUG

## FileAccessExtensionParams

::: health_universe_a2a.FileAccessExtensionParams
    options:
      show_root_heading: true

## FileAccessExtensionContext

::: health_universe_a2a.FileAccessExtensionContext
    options:
      show_root_heading: true

## BackgroundJobExtensionParams

::: health_universe_a2a.BackgroundJobExtensionParams
    options:
      show_root_heading: true

## BackgroundJobExtensionResponse

::: health_universe_a2a.BackgroundJobExtensionResponse
    options:
      show_root_heading: true

## BackgroundTaskResults

::: health_universe_a2a.BackgroundTaskResults
    options:
      show_root_heading: true

## Helper Functions

### ack_background_job_enqueued

::: health_universe_a2a.ack_background_job_enqueued

### notify_on_task_completion

::: health_universe_a2a.notify_on_task_completion

## Validation Types

### ValidationResult

::: health_universe_a2a.ValidationResult
    options:
      show_root_heading: true

### ValidationAccepted

::: health_universe_a2a.ValidationAccepted
    options:
      show_root_heading: true

### ValidationRejected

::: health_universe_a2a.ValidationRejected
    options:
      show_root_heading: true
