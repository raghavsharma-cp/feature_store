"""
Patients repository: insert into feature_store table.
Maps API patient object to table columns; unmapped columns stay null.
"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from sqlalchemy.orm import Session

from api.db.models import FeatureStore
from api.schemas.patients import PatientInput


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse ISO date-time string to date. Returns None if invalid or None."""
    if value is None:
        return None
    try:
        if "T" in value:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.date()
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _to_int(value: Any) -> Optional[int]:
    """Coerce to int for DB. Returns None if value is None or invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_str(value: Any) -> Optional[str]:
    """Coerce to str for DB (e.g. MRN, CPMRN). Returns None if value is None."""
    if value is None:
        return None
    return str(value).strip() or None


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Coerce to Decimal for DB. Returns None if value is None or invalid."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        return None


def patient_to_row(patient: PatientInput) -> FeatureStore:
    """Build a FeatureStore row from API patient input. Unmapped fields stay null."""
    row = FeatureStore(patient_id=patient._id)

    row.mrn = _to_str(patient.MRN)
    row.cpmrn = _to_str(patient.CPMRN)
    row.encounters = patient.encounters
    row.name = patient.name
    row.last_name = patient.lastName
    row.hospital_name = patient.hospitalName
    row.unit_name = _to_int(patient.unitName)
    row.bed_no = _to_int(patient.bedNo)
    row.camera = _to_int(patient.camera)
    row.hospital_id = patient.hospitalID
    row.unit_id = patient.unitID
    row.command_center_id = patient.commandCenterID

    if patient.age:
        row.age_year = _to_int(patient.age.year)
        row.age_month = _to_int(patient.age.month)
        row.age_days = _to_int(patient.age.days) if hasattr(patient.age, "days") else None
        row.age_hour = _to_int(patient.age.hour)
        row.age_day = _to_int(patient.age.day)
        row.age_minute = _to_int(patient.age.minute)

    row.dob = _parse_date(patient.dob)
    row.sex = patient.sex
    row.weight_unit = patient.weightUnit
    row.height_cm = _to_decimal(patient.height)
    row.weight_kg = _to_decimal(patient.weight)
    row.ibw = _to_decimal(patient.IBW)
    row.birth_weight = _to_decimal(patient.birthWeight)
    row.birth_weight_unit = patient.birthWeightUnit

    if patient.gestationAge:
        row.gestation_age_weeks = _to_int(patient.gestationAge.weeks)
        row.gestation_age_days = _to_int(patient.gestationAge.days)

    if patient.motherInfo and isinstance(patient.motherInfo, dict):
        row.mother_info_name = patient.motherInfo.get("name")
        row.mother_info_last_name = patient.motherInfo.get("lastName")
        row.mother_info_blood_group = patient.motherInfo.get("bloodGroup")
        row.mother_info_lmp = patient.motherInfo.get("lmp")
        ed = patient.motherInfo.get("expectedDelivery")
        row.mother_info_expected_delivery = (
            _parse_date(ed) if isinstance(ed, str) else None
        )

    row.icu_admit_date = _parse_date(patient.ICUAdmitDate)
    row.is_currently_admitted = patient.isCurrentlyAdmitted
    row.icu_discharge_date = _parse_date(patient.ICUDischargeDate)
    if patient.days is not None:
        row.days = len(patient.days) if isinstance(patient.days, list) else _to_int(patient.days)

    row.isolation = (
        patient.isolation if isinstance(patient.isolation, (list, dict)) else None
    )
    row.initial_symtoms = (
        patient.initialSymtoms
        if isinstance(patient.initialSymtoms, (list, dict))
        else None
    )
    row.allergies = (
        patient.allergies if isinstance(patient.allergies, (list, dict)) else None
    )
    row.chief_complaint = (
        patient.chiefComplaint
        if isinstance(patient.chiefComplaint, (list, dict))
        else None
    )
    row.other_complications = (
        patient.otherComplications
        if isinstance(patient.otherComplications, (list, dict))
        else None
    )
    row.signs_at_admission = (
        patient.signsAtAdmission
        if isinstance(patient.signsAtAdmission, (list, dict))
        else None
    )
    row.symptoms_at_admission = (
        patient.symptomsAtAdmission
        if isinstance(patient.symptomsAtAdmission, (list, dict))
        else None
    )
    row.past_medical_history = (
        patient.pastMedicalHistory
        if isinstance(patient.pastMedicalHistory, (dict, list))
        else (
            {"text": patient.pastMedicalHistory}
            if patient.pastMedicalHistory is not None
            else None
        )
    )
    row.diagnoses = (
        patient.diagnoses if isinstance(patient.diagnoses, (list, dict)) else None
    )

    return row


def mongo_patient_to_row(pat: dict) -> FeatureStore:
    """Build a FeatureStore row from a MongoDB patient document (camelCase keys)."""
    patient_id = str(pat.get("_id", ""))
    row = FeatureStore(patient_id=patient_id)

    row.mrn = _to_str(pat.get("MRN"))
    row.cpmrn = _to_str(pat.get("CPMRN"))
    row.encounters = pat.get("encounters")
    row.name = pat.get("name")
    row.last_name = pat.get("lastName")
    row.hospital_name = pat.get("hospitalName")
    row.unit_name = _to_int(pat.get("unitName"))
    row.bed_no = _to_int(pat.get("bedNo"))
    row.camera = _to_int(pat.get("camera"))
    row.hospital_id = str(pat.get("hospitalID", "")) if pat.get("hospitalID") else None
    row.unit_id = str(pat.get("unitID", "")) if pat.get("unitID") else None
    row.command_center_id = (
        str(pat.get("commandCenterID", "")) if pat.get("commandCenterID") else None
    )

    age = pat.get("age")
    if isinstance(age, dict):
        row.age_year = _to_int(age.get("year"))
        row.age_month = _to_int(age.get("month"))
        row.age_days = _to_int(age.get("days"))
        row.age_hour = _to_int(age.get("hour"))
        row.age_day = _to_int(age.get("day"))
        row.age_minute = _to_int(age.get("minute"))

    row.dob = _parse_date(pat.get("dob"))
    row.sex = pat.get("sex")
    row.weight_unit = pat.get("weightUnit")
    row.height_cm = _to_decimal(pat.get("heightCm") or pat.get("height"))
    row.weight_kg = _to_decimal(pat.get("weightKg") or pat.get("weight"))
    row.ibw = _to_decimal(pat.get("IBW"))
    row.birth_weight = _to_decimal(pat.get("birthWeight"))
    row.birth_weight_unit = pat.get("birthWeightUnit")

    ga = pat.get("gestationAge")
    if isinstance(ga, dict):
        row.gestation_age_weeks = _to_int(ga.get("weeks"))
        row.gestation_age_days = _to_int(ga.get("days"))

    mi = pat.get("motherInfo")
    if isinstance(mi, dict):
        row.mother_info_name = mi.get("name")
        row.mother_info_last_name = mi.get("lastName")
        row.mother_info_blood_group = mi.get("bloodGroup")
        row.mother_info_lmp = mi.get("lmp")
        ed = mi.get("expectedDelivery")
        row.mother_info_expected_delivery = (
            _parse_date(ed) if isinstance(ed, str) else None
        )

    row.icu_admit_date = _parse_date(pat.get("ICUAdmitDate"))
    row.is_currently_admitted = pat.get("isCurrentlyAdmitted")
    row.icu_discharge_date = _parse_date(pat.get("ICUDischargeDate"))
    days_val = pat.get("days")
    if days_val is not None:
        row.days = (
            len(days_val) if isinstance(days_val, list) else _to_int(days_val)
        )

    row.vent_free_days = _to_int(pat.get("ventFreeDays"))
    row.apache_score = _to_decimal(pat.get("apacheScore"))
    row.isolation = (
        pat.get("isolation")
        if isinstance(pat.get("isolation"), (list, dict))
        else None
    )
    row.initial_symtoms = (
        pat.get("initialSymtoms")
        if isinstance(pat.get("initialSymtoms"), (list, dict))
        else None
    )
    row.allergies = (
        pat.get("allergies")
        if isinstance(pat.get("allergies"), (list, dict))
        else None
    )
    row.chief_complaint = (
        pat.get("chiefComplaint")
        if isinstance(pat.get("chiefComplaint"), (list, dict))
        else None
    )
    row.other_complications = (
        pat.get("otherComplications")
        if isinstance(pat.get("otherComplications"), (list, dict))
        else None
    )
    row.signs_at_admission = (
        pat.get("signsAtAdmission")
        if isinstance(pat.get("signsAtAdmission"), (list, dict))
        else None
    )
    row.symptoms_at_admission = (
        pat.get("symptomsAtAdmission")
        if isinstance(pat.get("symptomsAtAdmission"), (list, dict))
        else None
    )
    row.underlying_medical_conditions = (
        pat.get("underlyingMedicalConditions")
        if isinstance(pat.get("underlyingMedicalConditions"), (list, dict))
        else None
    )
    row.chronic = (
        pat.get("chronic")
        if isinstance(pat.get("chronic"), (list, dict))
        else None
    )
    row.immune = (
        pat.get("immune")
        if isinstance(pat.get("immune"), (list, dict))
        else None
    )
    pmh = pat.get("pastMedicalHistory")
    row.past_medical_history = (
        pmh
        if isinstance(pmh, (dict, list))
        else ({"text": pmh} if pmh is not None else None)
    )
    row.diagnoses = (
        pat.get("diagnoses")
        if isinstance(pat.get("diagnoses"), (list, dict))
        else None
    )
    row.severity = pat.get("severity")
    row.transfer_history = (
        pat.get("transferHistory")
        if isinstance(pat.get("transferHistory"), (list, dict))
        else None
    )
    return row


def insert_patient(db: Session, patient: PatientInput) -> FeatureStore:
    """Insert a new row into feature_store. Returns the inserted row."""
    row = patient_to_row(patient)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
