"""
Shared schemas: pagination, errors.
"""

from pydantic import BaseModel


class PaginatedParams(BaseModel):
    """Common pagination query params."""

    skip: int = 0
    limit: int = 100


class ErrorDetail(BaseModel):
    """Error response detail."""

    detail: str
