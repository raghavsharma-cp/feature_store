"""
Health check router.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def health_check():
    """GET /health - basic liveness check."""
    return {"status": "ok"}
