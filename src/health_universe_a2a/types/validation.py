"""Validation types for message validation"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ValidationAccepted(BaseModel):
    """
    Message validation succeeded - agent will process the message.

    Attributes:
        status: Always "accepted"
        estimated_duration_seconds: Optional hint for estimated processing time
    """

    status: Literal["accepted"] = "accepted"
    estimated_duration_seconds: int | None = None


class ValidationRejected(BaseModel):
    """
    Message validation failed - agent will not process the message.

    Attributes:
        status: Always "rejected"
        reason: Human-readable reason for rejection
    """

    status: Literal["rejected"] = "rejected"
    reason: str = Field(..., description="Reason for rejecting the message")


# Discriminated union for type-safe validation results
ValidationResult = Annotated[
    ValidationAccepted | ValidationRejected,
    Field(discriminator="status"),
]
