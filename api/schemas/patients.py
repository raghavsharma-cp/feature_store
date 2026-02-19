"""
Patient request/response schemas.

Input matches the patient object (e.g. resultData from creatingNewPatient).
"""

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class AgeInput(BaseModel):
    """Age object in patient payload."""

    year: Optional[float] = None
    month: Optional[float] = None
    day: Optional[float] = None
    hour: Optional[float] = None
    minute: Optional[float] = None


class GestationAgeInput(BaseModel):
    """Gestation age in patient payload."""

    weeks: Optional[float] = None
    days: Optional[float] = None


class RequestInput(BaseModel):
    """Request metadata in patient payload."""

    type: Optional[Literal["GET"]] = None
    url: Optional[str] = None


class PatientInput(BaseModel):
    """Patient object returned by POST /patients/ (resultData from creatingNewPatient)."""

    model_config = ConfigDict(extra="allow")

    _id: str  # MongoDB ObjectId -> patient_id
    name: Optional[str] = None
    profilePicture: Optional[str] = None
    lastName: Optional[str] = None
    initialSymtoms: Optional[list[str]] = None
    allergies: Optional[list[str]] = None
    chiefComplaint: Optional[list[str]] = None
    otherComplications: Optional[list[str]] = None
    dateOfContactWithHealthFacility: Optional[str] = None
    dateOfHospitalization: Optional[str] = None
    dateOfOnsetSymptoms: Optional[str] = None
    ards: Optional[str] = None
    chestXray: Optional[str] = None
    renalFailure: Optional[str] = None
    coagulopathy: Optional[str] = None
    cardiacFailure: Optional[str] = None
    signsAtAdmission: Optional[list[str]] = None
    symptomsAtAdmission: Optional[list[str]] = None
    onSet: Optional[list[Any]] = None
    covidDetails: Optional[dict[str, Any]] = None
    covidInterviewerDetails: Optional[dict[str, Any]] = None
    occupation: Optional[str] = None
    nationality: Optional[str] = None
    fatherName: Optional[str] = None
    email: Optional[str] = None
    mobile: Optional[str] = None
    age: Optional[AgeInput] = None
    sex: Optional[str] = None
    lmp: Optional[str] = None
    phone: Optional[str] = None
    countryCode: Optional[str] = None
    patientImage: Optional[str] = None
    height: Optional[float] = None  # heightCm from DB
    heightUnit: Optional[str] = None
    weight: Optional[float] = None  # weightKg from DB
    weightObj: Optional[dict[str, Any]] = None
    weightUnit: Optional[str] = None
    birthWeight: Optional[float] = None
    birthWeightUnit: Optional[str] = None
    birthWeightObj: Optional[dict[str, Any]] = None
    IBW: Optional[float] = None
    dob: Optional[str] = None
    bloodGroup: Optional[str] = None
    MRN: Optional[str] = None
    patientType: Optional[Literal["adult", "neonatal", "pediatric"]] = None
    motherInfo: Optional[dict[str, Any]] = None
    gestationAge: Optional[GestationAgeInput] = None
    CPMRN: Optional[str] = None
    bedNo: Optional[str] = None
    camera: Optional[str] = None
    pastMedicalHistory: Optional[str] = None
    hospitalName: Optional[str] = None
    hospitalID: Optional[str] = None
    hospitalLogo: Optional[str] = None
    unitName: Optional[str] = None
    unitID: Optional[str] = None
    ABHA_ID: Optional[str] = None
    payorDetails: Optional[dict[str, Any]] = None
    aadhar: Optional[float] = None
    address: Optional[dict[str, Any]] = None
    covid: Optional[str] = None
    isolation: Optional[list[dict[str, Any]]] = None
    code: Optional[str] = None
    ICUAdmitDate: Optional[str] = None
    diagnoses: Optional[list[str]] = None
    MEWS: Optional[float] = None
    PCP: Optional[str] = None
    isNewPatient: Optional[Literal[True]] = None
    PCP_phone: Optional[str] = None
    PCP_speciality: Optional[str] = None
    PCP_email: Optional[str] = None
    PCP_ISDCode: Optional[str] = None
    isCurrentlyAdmitted: Optional[bool] = None
    ICUDischargeDate: Optional[str] = None
    ICUDischargeDisposition: Optional[str] = None
    ICUDischargeReason: Optional[str] = None
    days: Optional[list[Any]] = None
    encounters: Optional[int] = None
    createdBy: Optional[str] = None
    request: Optional[RequestInput] = None
    commandCenterID: Optional[str] = None
    visitId: Optional[str] = None


class AddNewPatientRequest(BaseModel):
    """Request body for addNewPatient: a single patient object."""

    patient: PatientInput
