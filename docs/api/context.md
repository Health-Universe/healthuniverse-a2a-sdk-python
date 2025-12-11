# Context API Reference

## BaseContext

::: health_universe_a2a.BaseContext
    options:
      show_root_heading: true

## StreamingContext

::: health_universe_a2a.StreamingContext
    options:
      show_root_heading: true
      members:
        - message
        - task_id
        - context_id
        - storage
        - extensions
        - add_artifact

## BackgroundContext

::: health_universe_a2a.BackgroundContext
    options:
      show_root_heading: true
      members:
        - message
        - task_id
        - context_id
        - file_access_params
        - extensions
        - update_progress
        - update_progress_sync
        - add_artifact
        - add_artifact_sync
