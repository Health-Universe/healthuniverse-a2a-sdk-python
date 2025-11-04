"""A2A Extension type definitions and constants"""

from pydantic import BaseModel

# Common extension URIs for Health Universe platform
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
