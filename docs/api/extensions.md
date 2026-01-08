# Extensions API Reference

## UpdateImportance

Importance levels for progress updates. Use these to control which updates are pushed to the Navigator UI.

::: health_universe_a2a.UpdateImportance
    options:
      show_root_heading: true
      show_source: true

## Validation Types

### ValidationAccepted

Return this from `validate_message` to accept the message for processing.

::: health_universe_a2a.ValidationAccepted
    options:
      show_root_heading: true
      show_source: true

### ValidationRejected

Return this from `validate_message` to reject the message with a reason.

::: health_universe_a2a.ValidationRejected
    options:
      show_root_heading: true
      show_source: true
