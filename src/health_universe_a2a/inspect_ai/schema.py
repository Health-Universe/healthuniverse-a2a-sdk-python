"""
Pydantic schema models for Inspect AI .eval file format.

These models validate the structure of eval files to ensure compatibility
with the `inspect view` UI. Use these to validate data before writing.
"""

from typing import Any

from pydantic import BaseModel, Field


class ModelUsageEntry(BaseModel):
    """Token usage for a single model."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_cache_read: int | None = None
    reasoning_tokens: int | None = None


class EvalDataset(BaseModel):
    """Dataset configuration in eval spec."""

    name: str
    samples: int
    sample_ids: list[str]
    shuffled: bool = False


class EvalSpec(BaseModel):
    """Evaluation specification - the 'eval' field in header.json."""

    eval_id: str
    run_id: str
    created: str
    task: str
    task_id: str
    task_version: int = 1
    task_attribs: dict[str, Any] = Field(default_factory=dict)
    task_args: dict[str, Any] = Field(default_factory=dict)
    task_args_passed: dict[str, Any] = Field(default_factory=dict)
    model: str
    model_generate_config: dict[str, Any] = Field(default_factory=dict)
    model_args: dict[str, Any] = Field(default_factory=dict)
    dataset: EvalDataset
    config: dict[str, Any] = Field(default_factory=dict)
    packages: dict[str, str] = Field(default_factory=dict)


class EvalPlan(BaseModel):
    """Evaluation plan configuration."""

    name: str
    steps: list[Any] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalResults(BaseModel):
    """Evaluation results summary."""

    total_samples: int
    completed_samples: int
    scores: list[Any] = Field(default_factory=list)


class EvalStats(BaseModel):
    """Evaluation statistics including model usage."""

    started_at: str
    completed_at: str
    model_usage: dict[str, ModelUsageEntry]


class EvalHeader(BaseModel):
    """
    Complete header.json structure.

    This is the main metadata file in an .eval archive.
    All fields are required by the inspect view UI.
    """

    version: int = 2
    status: str  # "success" or "error"
    eval: EvalSpec
    plan: EvalPlan
    results: EvalResults
    stats: EvalStats
    error: dict[str, str] | None = None


class SampleOutput(BaseModel):
    """Sample output configuration."""

    model: str
    choices: list[Any] = Field(default_factory=list)
    text: str | None = None


class EvalEvent(BaseModel):
    """Base event in sample events list."""

    event: str
    timestamp: str
    # Additional fields vary by event type


class EvalSample(BaseModel):
    """
    Sample data structure for samples/1_epoch_1.json.

    Contains the full execution data including all events.
    """

    id: str
    epoch: int = 1
    input: str
    target: str = ""
    messages: list[Any] = Field(default_factory=list)
    output: SampleOutput
    events: list[dict[str, Any]] = Field(default_factory=list)
    model_usage: dict[str, ModelUsageEntry]
    metadata: dict[str, Any] = Field(default_factory=dict)
    store: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    completed_at: str
    total_time: float
    working_time: float
    uuid: str
    scores: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class SampleSummary(BaseModel):
    """
    Sample summary for summaries.json.

    summaries.json MUST be a list of these objects, not a single object.
    The inspect view UI calls .map() on this array.
    """

    id: int
    epoch: int = 1
    input: str
    target: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    model_usage: dict[str, ModelUsageEntry]
    started_at: str
    completed_at: str
    total_time: float
    working_time: float
    uuid: str
    retries: int = 0
    completed: bool
    message_count: int = 0


class JournalStart(BaseModel):
    """Journal start metadata for _journal/start.json."""

    eval_id: str
    run_id: str
    task: str
    model: str
    created: str


def validate_eval_header(data: dict[str, Any]) -> EvalHeader:
    """Validate header.json data and return typed model."""
    return EvalHeader(**data)


def validate_eval_sample(data: dict[str, Any]) -> EvalSample:
    """Validate sample data and return typed model."""
    return EvalSample(**data)


def validate_summaries(data: list[dict[str, Any]]) -> list[SampleSummary]:
    """Validate summaries.json data and return typed models."""
    return [SampleSummary(**s) for s in data]


def validate_journal_start(data: dict[str, Any]) -> JournalStart:
    """Validate journal start data and return typed model."""
    return JournalStart(**data)
