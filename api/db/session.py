"""
Database session: engine and session factory for PostgreSQL.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from api.config import get_settings

settings = get_settings()
engine = create_engine(
    settings.postgres_url,
    pool_pre_ping=True,
    echo=settings.api_env == "development",
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a DB session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
