"""A2A Extension type definitions"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class AgentExtension:
    """
    A2A Agent Extension declaration.

    Extensions are capabilities that agents can declare support for
    in their AgentCard. Common extensions include:
    - File Access: https://healthuniverse.com/ext/file_access/v1
    - Background Jobs: https://healthuniverse.com/ext/background_job/v1

    Attributes:
        uri: URI identifying the extension
        metadata: Optional metadata about extension support
    """

    uri: str
    metadata: dict[str, Any] | None = None


# Common extension URIs
FILE_ACCESS_EXTENSION_URI = "https://healthuniverse.com/ext/file_access/v1"
BACKGROUND_JOB_EXTENSION_URI = "https://healthuniverse.com/ext/background_job/v1"


# Extension parameter models


class BackgroundJobExtensionParams(BaseModel):
    """
    Parameters for Background Job extension.

    Passed in message metadata when using background job extension.

    Attributes:
        api_key: API key for POSTing updates to backend
        job_id: Unique job ID for tracking this background task
    """

    api_key: str
    job_id: str
