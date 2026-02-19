"""
Patients API router.
"""

from fastapi import APIRouter, Depends

from api.schemas.patients import AddNewPatientRequest
from api.services import patients_service

router = APIRouter()


@router.post("/addNewPatient")
def add_new_patient(
    payload: AddNewPatientRequest,
    service: patients_service.PatientsService = Depends(
        patients_service.get_patients_service
    ),
):
    """Insert patient into feature_store table. Request body: { \"patient\": <PatientInput> }."""
    return service.add_new_patient(patient=payload.patient)
