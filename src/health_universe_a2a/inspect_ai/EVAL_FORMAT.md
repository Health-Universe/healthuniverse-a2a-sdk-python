# Inspect AI .eval File Format

This document describes the `.eval` file format used by Inspect AI for viewing agent execution logs.

## Overview

`.eval` files are **Zip archives** containing JSON files that describe an evaluation run. The `inspect view` command reads these files to display execution details in a web UI.

## File Structure

```
example.eval (Zip archive)
├── header.json           # Evaluation metadata and stats
├── samples/
│   └── 1_epoch_1.json    # Sample execution data with events
├── summaries.json        # Array of sample summaries
├── _journal/
│   ├── start.json        # Journal start metadata
│   └── summaries/
│       └── 1.json        # Journal summary (same as summaries.json)
```

## Required Files

### header.json

Contains evaluation metadata, configuration, and statistics.

```json
{
  "version": 2,
  "status": "success",
  "eval": {
    "eval_id": "unique-eval-id",
    "run_id": "unique-run-id",
    "created": "2024-01-01T00:00:00+00:00",
    "task": "task_name",
    "task_id": "task_name-eval_id",
    "task_version": 1,
    "task_attribs": {},
    "task_args": {},
    "task_args_passed": {},
    "model": "model-name",
    "model_generate_config": {},
    "model_args": {},
    "dataset": {
      "name": "dataset_name",
      "samples": 1,
      "sample_ids": ["sample-uuid"],
      "shuffled": false
    },
    "config": {},
    "packages": {"package_name": "version"}
  },
  "plan": {
    "name": "plan_name",
    "steps": [],
    "config": {}
  },
  "results": {
    "total_samples": 1,
    "completed_samples": 1,
    "scores": []
  },
  "stats": {
    "started_at": "2024-01-01T00:00:00+00:00",
    "completed_at": "2024-01-01T00:00:01+00:00",
    "model_usage": {
      "model-name": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
      }
    }
  }
}
```

### samples/1_epoch_1.json

Contains the sample execution data including all events.

```json
{
  "id": "sample-uuid",
  "epoch": 1,
  "input": "Input description",
  "target": "",
  "messages": [],
  "output": {
    "model": "model-name",
    "choices": []
  },
  "events": [
    {
      "event": "info",
      "timestamp": "2024-01-01T00:00:00.000000+00:00",
      "source": "source_name",
      "data": {}
    }
  ],
  "model_usage": {
    "model-name": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  },
  "metadata": {},
  "store": {},
  "started_at": "2024-01-01T00:00:00+00:00",
  "completed_at": "2024-01-01T00:00:01+00:00",
  "total_time": 1.0,
  "working_time": 0.5,
  "uuid": "sample-uuid"
}
```

### summaries.json

Array of sample summaries for the UI to iterate over.

```json
[
  {
    "id": 1,
    "epoch": 1,
    "input": "Input description",
    "target": "",
    "metadata": {},
    "scores": {},
    "model_usage": {
      "model-name": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
      }
    },
    "started_at": "2024-01-01T00:00:00+00:00",
    "completed_at": "2024-01-01T00:00:01+00:00",
    "total_time": 1.0,
    "working_time": 0.5,
    "uuid": "sample-uuid",
    "retries": 0,
    "completed": true,
    "message_count": 0
  }
]
```

### _journal/start.json

Journal start metadata.

```json
{
  "eval_id": "unique-eval-id",
  "run_id": "unique-run-id",
  "task": "task_name",
  "model": "model-name",
  "created": "2024-01-01T00:00:00+00:00"
}
```

## Critical Requirements

### Fields Required by inspect view UI

The following fields **must be present** (even if empty) or the UI will crash:

| Field | Location | Required Value |
|-------|----------|----------------|
| `model_generate_config` | header.json → eval | `{}` (empty object) |
| `model_args` | header.json → eval | `{}` (empty object) |
| `task_attribs` | header.json → eval | `{}` (empty object) |
| `task_args` | header.json → eval | `{}` (empty object) |
| `task_args_passed` | header.json → eval | `{}` (empty object) |
| `packages` | header.json → eval | `{}` or `{"pkg": "ver"}` |

### model_usage Structure

The `model_usage` field **must be keyed by model name**:

```json
// ✓ Correct
"model_usage": {
  "gpt-4": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
}

// ✗ Wrong - will crash Models tab
"model_usage": {}

// ✗ Wrong - will crash Models tab
"model_usage": {"input_tokens": 100, "output_tokens": 50}
```

### summaries.json Must Be Array

```json
// ✓ Correct
[{"id": 1, "epoch": 1, ...}]

// ✗ Wrong - will crash with "map is not a function"
{"samples": 1, "completed": 1}
```

## Event Types

Events in `samples/1_epoch_1.json` can have these types:

| Event Type | Description |
|------------|-------------|
| `info` | General information event |
| `tool` | Tool/function call |
| `model` | Model inference |
| `sample_init` | Sample initialization |
| `state` | State change |
| `subtask` | Subtask execution |
| `span_begin` | Start of a span |
| `span_end` | End of a span |

### Tool Event Example

```json
{
  "event": "tool",
  "timestamp": "2024-01-01T00:00:00.000000+00:00",
  "type": "function",
  "id": "call_123",
  "function": "tool_name",
  "arguments": {"param": "value"},
  "result": {"success": true},
  "events": [],
  "view": "markdown",
  "truncated": null
}
```

### Info Event Example

```json
{
  "event": "info",
  "timestamp": "2024-01-01T00:00:00.000000+00:00",
  "source": "a2a:task",
  "data": {"state": "working", "message": "Processing..."}
}
```

## Validation

Use the `InspectEvalSchema` Pydantic models in `schema.py` to validate eval files before writing:

```python
from health_universe_a2a.inspect_ai.schema import EvalHeader, SampleSummary

# Validate header
header = EvalHeader(**header_dict)

# Validate summaries
summaries = [SampleSummary(**s) for s in summaries_list]
```
