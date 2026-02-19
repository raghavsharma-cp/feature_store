"""
Patients business logic: orchestrates use cases and calls db repositories.
"""

from api.db.repositories import patients as patients_repo
from api.db.session import get_db
from api.schemas.patients import PatientInput
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session


class PatientsService:
    """Service for patient-related use cases."""

    def __init__(self, db: Session):
        self._db = db

    def add_new_patient(self, patient: PatientInput) -> dict:
        """Insert a new patient row into feature_store. Returns status and patient_id."""
        row = patients_repo.insert_patient(self._db, patient)
        return {"status": "ok", "patient_id": row.patient_id}


def get_patients_service(db: Session = Depends(get_db)) -> PatientsService:
    """Dependency that provides PatientsService with a DB session."""
    return PatientsService(db)
