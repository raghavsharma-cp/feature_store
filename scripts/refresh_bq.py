"""
Refresh BigQuery feature store tables from MongoDB for patients discharged in the last 2 months.

Mirrors refresh_data.py flow but writes to BQ only (no Postgres).
Uses get_discharged_patients_in_range(start_date, end_date) with last 2 months.

Usage (from feature_store project root):
  python -m scripts.refresh_bq
  PYTHONPATH=. python scripts/refresh_bq.py

Requires: db_uri, BQ_PROJECT_ID, BQ_DATASET_ID in env or .env / .env.local
"""

import logging
import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            if "T" in value:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.date()
            return date.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, (int, float)):
        try:
            if value > 1e10:
                value = value / 1000.0
            dt = datetime.fromtimestamp(value)
            return dt.date()
        except (ValueError, TypeError, OSError):
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return None


def _bq_val(v: Any) -> Any:
    """Serialize for BQ JSON insert: date -> str, Decimal -> float."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _mongo_patient_to_dict(pat: dict) -> Dict[str, Any]:
    """Build feature_store row as dict (BQ column names)."""
    patient_id = str(pat.get("_id", ""))
    row = {"patient_id": patient_id}
    row["mrn"] = str(pat.get("MRN")) if pat.get("MRN") is not None else None
    row["cpmrn"] = str(pat.get("CPMRN")) if pat.get("CPMRN") is not None else None
    row["encounters"] = pat.get("encounters")
    row["name"] = pat.get("name")
    row["last_name"] = pat.get("lastName")
    row["hospital_name"] = pat.get("hospitalName")
    row["unit_name"] = _to_int(pat.get("unitName"))
    row["bed_no"] = _to_int(pat.get("bedNo"))
    row["camera"] = _to_int(pat.get("camera"))
    row["hospital_id"] = str(pat.get("hospitalID")) if pat.get("hospitalID") else None
    row["unit_id"] = str(pat.get("unitID")) if pat.get("unitID") else None
    row["command_center_id"] = str(pat.get("commandCenterID")) if pat.get("commandCenterID") else None
    age = pat.get("age")
    if isinstance(age, dict):
        row["age_year"] = _to_int(age.get("year"))
        row["age_month"] = _to_int(age.get("month"))
        row["age_days"] = _to_int(age.get("days"))
        row["age_hour"] = _to_int(age.get("hour"))
        row["age_day"] = _to_int(age.get("day"))
        row["age_minute"] = _to_int(age.get("minute"))
    else:
        row["age_year"] = row["age_month"] = row["age_days"] = row["age_hour"] = row["age_day"] = row["age_minute"] = None
    row["dob"] = _parse_date(pat.get("dob"))
    row["sex"] = pat.get("sex")
    row["weight_unit"] = pat.get("weightUnit")
    row["height_cm"] = _to_decimal(pat.get("heightCm") or pat.get("height"))
    row["weight_kg"] = _to_decimal(pat.get("weightKg") or pat.get("weight"))
    row["ibw"] = _to_decimal(pat.get("IBW"))
    row["bmi"] = _to_decimal(pat.get("BMI"))
    row["birth_weight"] = _to_decimal(pat.get("birthWeight"))
    row["birth_weight_unit"] = pat.get("birthWeightUnit")
    ga = pat.get("gestationAge")
    if isinstance(ga, dict):
        row["gestation_age_weeks"] = _to_int(ga.get("weeks"))
        row["gestation_age_days"] = _to_int(ga.get("days"))
    else:
        row["gestation_age_weeks"] = row["gestation_age_days"] = None
    mi = pat.get("motherInfo")
    if isinstance(mi, dict):
        row["mother_info_name"] = mi.get("name")
        row["mother_info_last_name"] = mi.get("lastName")
        row["mother_info_blood_group"] = mi.get("bloodGroup")
        row["mother_info_lmp"] = mi.get("lmp")
        ed = mi.get("expectedDelivery")
        row["mother_info_expected_delivery"] = _parse_date(ed) if isinstance(ed, str) else None
    else:
        row["mother_info_name"] = row["mother_info_last_name"] = row["mother_info_blood_group"] = row["mother_info_lmp"] = row["mother_info_expected_delivery"] = None
    row["weight_history"] = pat.get("weightHistory") if isinstance(pat.get("weightHistory"), (list, dict)) else None
    row["icu_admit_date"] = _parse_date(pat.get("ICUAdmitDate"))
    row["is_currently_admitted"] = pat.get("isCurrentlyAdmitted")
    row["icu_discharge_date"] = _parse_date(pat.get("ICUDischargeDate"))
    days_val = pat.get("days")
    row["days"] = len(days_val) if isinstance(days_val, list) else _to_int(days_val) if days_val is not None else None
    row["vent_free_days"] = _to_int(pat.get("ventFreeDays"))
    row["apache_score"] = _to_decimal(pat.get("apacheScore"))
    row["isolation"] = pat.get("isolation") if isinstance(pat.get("isolation"), (list, dict)) else None
    row["initial_symtoms"] = pat.get("initialSymtoms") if isinstance(pat.get("initialSymtoms"), (list, dict)) else None
    row["allergies"] = pat.get("allergies") if isinstance(pat.get("allergies"), (list, dict)) else None
    row["chief_complaint"] = pat.get("chiefComplaint") if isinstance(pat.get("chiefComplaint"), (list, dict)) else None
    row["other_complications"] = pat.get("otherComplications") if isinstance(pat.get("otherComplications"), (list, dict)) else None
    row["signs_at_admission"] = pat.get("signsAtAdmission") if isinstance(pat.get("signsAtAdmission"), (list, dict)) else None
    row["symptoms_at_admission"] = pat.get("symptomsAtAdmission") if isinstance(pat.get("symptomsAtAdmission"), (list, dict)) else None
    row["underlying_medical_conditions"] = pat.get("underlyingMedicalConditions") if isinstance(pat.get("underlyingMedicalConditions"), (list, dict)) else None
    row["chronic"] = pat.get("chronic") if isinstance(pat.get("chronic"), (list, dict)) else None
    row["immune"] = pat.get("immune") if isinstance(pat.get("immune"), (list, dict)) else None
    pmh = pat.get("pastMedicalHistory")
    row["past_medical_history"] = pmh if isinstance(pmh, (dict, list)) else ({"text": pmh} if pmh is not None else None)
    row["diagnoses"] = pat.get("diagnoses") if isinstance(pat.get("diagnoses"), (list, dict)) else None
    row["transfer_history"] = pat.get("transferHistory") if isinstance(pat.get("transferHistory"), (list, dict)) else None
    row["documents"] = row["vitals"] = row["abnormal_vitals"] = None
    row["severity"] = pat.get("severity")
    row["orders"] = row["completed_orders"] = row["notes"] = None
    row["notes_diagnoses"] = row["notes_symptoms"] = row["notes_medications"] = row["notes_procedures"] = row["notes_tests"] = None
    return row


def _mongo_vital_to_dict(vital: dict, patient_id: str, vital_id: str) -> Dict[str, Any]:
    _abnormal_type_to_col = {
        "HR": "hr", "RR": "rr", "BP": "bp", "MAP": "map", "CVP": "cvp",
        "SPO2": "sp_o2", "FIO2": "fi_o2", "TEMPERATURE": "temperature",
        "TEMP": "temperature", "AVPU": "avpu", "PATPOSITION": "pat_position",
        "POSITION": "pat_position",
    }
    _valid_abnormal_cols = {"hr", "rr", "bp", "map", "cvp", "sp_o2", "fi_o2", "temperature", "avpu", "pat_position"}
    ab_list = vital.get("abnormal_list")
    if isinstance(ab_list, list):
        col_names = []
        for x in ab_list:
            if not isinstance(x, dict) or not x.get("type"):
                continue
            type_key = str(x.get("type")).strip().upper().replace(" ", "")
            col = _abnormal_type_to_col.get(type_key)
            if col and col not in col_names:
                col_names.append(col)
        abnormal_fields = col_names if col_names else None
    elif isinstance(vital.get("abnormal_fields"), list):
        abnormal_fields = [str(x) for x in vital["abnormal_fields"] if x is not None and str(x) in _valid_abnormal_cols] or None
    else:
        abnormal_fields = None
    return {
        "vital_id": vital_id,
        "patient_id": patient_id,
        "timestamp": _parse_date(vital.get("timestamp")),
        "therapy_device": vital.get("daysTherapyDevice") if isinstance(vital.get("daysTherapyDevice"), (list, dict)) else None,
        "vent_airway": vital.get("daysVentAirway") if isinstance(vital.get("daysVentAirway"), (list, dict)) else None,
        "temperature": _to_decimal(vital.get("daysTemperature")),
        "temperature_unit": vital.get("daysTemperatureUnit"),
        "avpu": vital.get("daysAVPU"),
        "hr": _to_int(vital.get("daysHR")),
        "rr": _to_int(vital.get("daysRR")),
        "bp": str(vital.get("daysBP")) if vital.get("daysBP") is not None else None,
        "reason_bp": vital.get("daysReasonBP"),
        "map": _to_decimal(vital.get("daysMAP")),
        "cvp": _to_decimal(vital.get("daysCVP")),
        "sp_o2": _to_int(vital.get("daysSpO2")),
        "fi_o2": _to_int(vital.get("daysFiO2")),
        "pat_position": vital.get("daysPatPosition"),
        "is_verified": vital.get("isVerified"),
        "verified_time": _parse_date(vital.get("verifiedTime")),
        "verified_by_name": vital.get("verifiedByName"),
        "verified_by_email": vital.get("verifiedByEmail"),
        "abnormal_fields": abnormal_fields,
    }


def _mongo_document_to_dict(doc: dict, patient_id: str, document_id: str) -> Dict[str, Any]:
    return {
        "document_id": document_id,
        "patient_id": patient_id,
        "is_inactive": doc.get("isInactive"),
        "tags": doc.get("tags") if isinstance(doc.get("tags"), (list, dict)) else None,
        "secondary_keys": doc.get("secondaryKeys") if isinstance(doc.get("secondaryKeys"), (list, dict)) else None,
        "snomed_code": doc.get("snomedCode") if isinstance(doc.get("snomedCode"), (list, dict)) else None,
        "key": doc.get("key"),
        "attributes": doc.get("attributes") if isinstance(doc.get("attributes"), (list, dict)) else None,
        "reported_at": _parse_date(doc.get("reportedAt")),
        "name": doc.get("name"),
        "verified_at": _parse_date(doc.get("verifiedAt")),
        "classifications": doc.get("classifications") if isinstance(doc.get("classifications"), (list, dict)) else None,
        "label": doc.get("label"),
    }


def _mongo_note_to_dict(
    note: dict, content: dict, component: dict,
    patient_id: str, note_id: str, content_id: str, component_id: str,
) -> Dict[str, Any]:
    author = content.get("author") or {}
    author_name = author.get("name") if isinstance(author, dict) else None
    author_email = author.get("email") if isinstance(author, dict) else None
    addendum = content.get("addendum")
    if isinstance(addendum, list) and addendum:
        last_add = addendum[-1]
        latest_addendum_author = last_add.get("name") if isinstance(last_add, dict) else None
        latest_addendum_timestamp = _parse_date(last_add.get("timestamp") if isinstance(last_add, dict) else None)
        latest_addendum_text = last_add.get("note") if isinstance(last_add, dict) else None
    else:
        latest_addendum_author = latest_addendum_timestamp = latest_addendum_text = None
    return {
        "patient_id": patient_id,
        "note_id": note_id,
        "content_id": content_id,
        "component_id": component_id,
        "created_timestamp": _parse_date(note.get("createdTimestamp") or note.get("createdAt")),
        "author": author_name,
        "author_id": str(author_email or ""),
        "author_role": author.get("role") if isinstance(author, dict) else None,
        "author_is_client": content.get("authorIsClient"),
        "note_type": content.get("noteType"),
        "note_sub_type": content.get("noteSubType"),
        "pend_or_signed": content.get("pendOrSigned"),
        "component_raw_value": component.get("value"),
        "component_parsed_text": None,
        "is_delete_enabled": note.get("isDeleteEnabled"),
        "chargeable": str(content.get("chargeable")) if content.get("chargeable") is not None else None,
        "impact_case": content.get("impactCase"),
        "addendum_json": content.get("addendum") if isinstance(content.get("addendum"), (list, dict)) else None,
        "latest_addendum_author": latest_addendum_author,
        "latest_addendum_timestamp": latest_addendum_timestamp,
        "latest_addendum_text": latest_addendum_text,
        "note_diagnoses": [],
        "note_symptoms": [],
        "note_medications": [],
        "note_procedures": [],
        "note_tests": [],
        "api_cost": 0,
    }


def _mongo_order_to_dict(order: dict, patient_id: str, order_id: str) -> Dict[str, Any]:
    def rate_val(o, key):
        r = o.get("rate")
        return r.get("value") if isinstance(r, dict) else None
    def rate_unit(o, key):
        r = o.get("rate")
        return r.get("unit") if isinstance(r, dict) else None
    return {
        "order_id": order_id,
        "patient_id": patient_id,
        "order_no": order.get("orderNo"),
        "lab_id": order.get("labId"),
        "category": order.get("category"),
        "type": order.get("type"),
        "state": order.get("state"),
        "created_at": _parse_date(order.get("createdAt")),
        "updated_at": _parse_date(order.get("updatedAt")),
        "signed": order.get("signed"),
        "preset": order.get("preset"),
        "orderable": order.get("orderable"),
        "protocol": order.get("protocol"),
        "history": order.get("history") if isinstance(order.get("history"), (list, dict)) else None,
        "additional_information": order.get("additionalInformation"),
        "sos": order.get("sos"),
        "sos_reason": order.get("sosReason"),
        "discontinue_reasons": order.get("discontinueReasons") if isinstance(order.get("discontinueReasons"), (list, dict)) else None,
        "discontinue_at": _parse_date(order.get("discontinueAt")),
        "discontinue_by": order.get("discontinueBy"),
        "completed_at": _parse_date(order.get("completedAt")),
        "completed_by": order.get("completedBy"),
        "urgency": order.get("urgency"),
        "start_time": _parse_date(order.get("startTime")),
        "frequency": order.get("frequency") if isinstance(order.get("frequency"), (list, dict)) else None,
        "number_of_doses": _to_int(order.get("numberOfDoses")),
        "medication_pta": order.get("medication_pta"),
        "medication_concentration": order.get("medication_concentration") if isinstance(order.get("medication_concentration"), (list, dict)) else None,
        "medication_combination": order.get("medication_combination") if isinstance(order.get("medication_combination"), (list, dict)) else None,
        "medication_name": order.get("medication_name") or order.get("name"),
        "medication_quantity": _to_int(order.get("medication_quantity") or order.get("quantity")),
        "medication_unit": order.get("medication_unit") or order.get("unit"),
        "medication_route": order.get("medication_route") or order.get("route"),
        "medication_no_of_days": _to_int(order.get("medication_noOfDays") or order.get("noOfDays")),
        "medication_form": order.get("medication_form") or order.get("form"),
        "medication_max_dose": _to_int(order.get("medication_maxDose") or order.get("maxDose")),
        "medication_body_weight": _to_int(order.get("medication_bodyWeight") or order.get("bodyWeight")),
        "labs_investigation": order.get("labs_investigation") or order.get("investigation"),
        "labs_discipline": order.get("labs_discipline") or order.get("discipline"),
        "labs_specimen_type": order.get("labs_specimenType") or order.get("specimenType"),
        "diets_name": order.get("diets_name") or order.get("name"),
        "diets_rate_value": _to_decimal(order.get("diets_rate_value") or (order.get("rate") or {}).get("value")),
        "diets_rate_unit": (order.get("rate") or {}).get("unit"),
        "communications_gcs": order.get("communications_gcs") or order.get("gcs"),
        "communications_title": order.get("communications_title") or order.get("title"),
        "bloods_start_now": order.get("bloods_startNow") or order.get("startNow"),
        "bloods_time_period_detail": order.get("bloods_timePeriodDetail") if isinstance(order.get("bloods_timePeriodDetail"), (list, dict)) else None,
        "bloods_skip_schedule": order.get("bloods_skipSchedule") if isinstance(order.get("bloods_skipSchedule"), (list, dict)) else None,
        "bloods_title": order.get("bloods_title"),
        "bloods_quantity": _to_int(order.get("bloods_quantity")),
        "bloods_quantity_unit": order.get("bloods_quantityUnit"),
        "bloods_schedule_selector": order.get("bloods_scheduleSelector"),
        "bloods_snomed_code": _to_int(order.get("bloods_snomedCode")),
        "procedure_name": order.get("procedure_name"),
        "procedure_ptype": order.get("procedure_ptype"),
        "procedure_laterality": order.get("procedure_laterality"),
        "procedure_site": order.get("procedure_site"),
    }


def _extract_orders_flat(orders_field: Any) -> List[tuple]:
    out: List[tuple] = []
    if not isinstance(orders_field, dict):
        return out
    for bucket in ("active", "pending", "completed"):
        bucket_data = orders_field.get(bucket)
        if not isinstance(bucket_data, dict):
            continue
        for key in ("medications", "labs", "diets", "bloods", "procedures", "vents", "communications"):
            items = bucket_data.get(key)
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                oid = item.get("_id") or item.get("id") or f"{bucket}_{key}_{i}"
                out.append((str(oid), item))
    return out


def _serialize_row_for_bq(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make dict BQ-insert safe: date -> str, Decimal -> float."""
    return {k: _bq_val(v) for k, v in d.items()}


def run_refresh_bq(limit: Optional[int] = None) -> None:
    from feature_store.feature_store import BaseFeatureStore, convert_to_serializable
    from scripts.bq_client import get_bq_client, get_dataset_id, get_table_ref
    from google.cloud import bigquery

    if not os.environ.get("db_uri"):
        logger.error("db_uri environment variable is not set")
        raise SystemExit(1)
    if not os.environ.get("BQ_PROJECT_ID") or not os.environ.get("BQ_DATASET_ID"):
        logger.error("BQ_PROJECT_ID and BQ_DATASET_ID are required")
        raise SystemExit(1)

    end_date = date.today()
    start_date = end_date - timedelta(days=60)  # ~2 months
    store = BaseFeatureStore()
    patients = store.get_discharged_patients_in_range(start_date, end_date, limit=limit)
    logger.info("Processing %d discharged patients for BQ", len(patients))

    bq_client = get_bq_client()
    dataset_id = get_dataset_id()
    project_id = bq_client.project
    full_dataset = f"{project_id}.{dataset_id}"

    all_feature_store = []
    all_vitals = []
    all_documents = []
    all_notes = []
    all_orders = []
    patient_ids = []

    for pat in patients:
        serializable = convert_to_serializable(pat)
        if not isinstance(serializable, dict):
            continue
        patient_id = str(serializable.get("_id", ""))
        patient_ids.append(patient_id)

        fs_row = _mongo_patient_to_dict(serializable)
        signed_order_ids = []
        for oid, o in _extract_orders_flat(serializable.get("orders")):
            if not o.get("signed") or (o.get("category") and str(o.get("category")).strip().lower() == "pending"):
                continue
            signed_order_ids.append(oid)
            all_orders.append(_mongo_order_to_dict(o, patient_id, oid))
        fs_row["orders"] = signed_order_ids or None
        all_feature_store.append(fs_row)

        for idx, v in enumerate(serializable.get("vitals") or []):
            if not isinstance(v, dict):
                continue
            vid = str(v.get("_id") or f"{patient_id}_vital_{idx}")
            all_vitals.append(_mongo_vital_to_dict(v, patient_id, vid))
        for idx, d in enumerate(serializable.get("documents") or []):
            if not isinstance(d, dict):
                continue
            doc_id = str(d.get("_id") or d.get("id") or f"{patient_id}_doc_{idx}")
            all_documents.append(_mongo_document_to_dict(d, patient_id, doc_id))
        notes_data = serializable.get("notes")
        if isinstance(notes_data, dict):
            notes_list = notes_data.get("finalNotes") or []
        else:
            notes_list = notes_data if isinstance(notes_data, list) else []
        for idx, n in enumerate(notes_list):
            if not isinstance(n, dict):
                continue
            note_id = str(n.get("_id") or n.get("id") or f"{patient_id}_note_{idx}")
            for cidx, content in enumerate(n.get("content") or []):
                if not isinstance(content, dict):
                    continue
                content_id = str(content.get("_id") or content.get("id") or f"{note_id}_c_{cidx}")
                for comp_idx, component in enumerate(content.get("components") or []):
                    if not isinstance(component, dict):
                        continue
                    comp_id = str(component.get("id") or f"_{comp_idx}")
                    all_notes.append(_mongo_note_to_dict(n, content, component, patient_id, note_id, content_id, comp_id))

    if not patient_ids:
        logger.info("No rows to write to BQ")
        return

    # Delete existing rows for these patient_ids in BQ (idempotent refresh), then insert
    tables_with_patient_id = [
        "vitals_feature_store",
        "documents_feature_store",
        "notes_feature_store",
        "orders_feature_store",
        "feature_store",
    ]
    for table_name in tables_with_patient_id:
        table_ref = f"{full_dataset}.{table_name}"
        for i in range(0, len(patient_ids), 500):
            batch = patient_ids[i : i + 500]
            placeholders = ", ".join([f"'{str(p).replace(chr(39), chr(39)+chr(39))}'" for p in batch])
            q = f"DELETE FROM `{table_ref}` WHERE patient_id IN ({placeholders})"
            bq_client.query(q)

    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND, autodetect=False)
    if all_vitals:
        bq_client.load_table_from_json(
            [_serialize_row_for_bq(r) for r in all_vitals],
            f"{full_dataset}.vitals_feature_store",
            job_config=job_config,
        ).result()
        logger.info("Loaded %d vitals", len(all_vitals))
    if all_documents:
        bq_client.load_table_from_json(
            [_serialize_row_for_bq(r) for r in all_documents],
            f"{full_dataset}.documents_feature_store",
            job_config=job_config,
        ).result()
        logger.info("Loaded %d documents", len(all_documents))
    if all_notes:
        bq_client.load_table_from_json(
            [_serialize_row_for_bq(r) for r in all_notes],
            f"{full_dataset}.notes_feature_store",
            job_config=job_config,
        ).result()
        logger.info("Loaded %d notes", len(all_notes))
    if all_orders:
        bq_client.load_table_from_json(
            [_serialize_row_for_bq(r) for r in all_orders],
            f"{full_dataset}.orders_feature_store",
            job_config=job_config,
        ).result()
        logger.info("Loaded %d orders", len(all_orders))
    if all_feature_store:
        bq_client.load_table_from_json(
            [_serialize_row_for_bq(r) for r in all_feature_store],
            f"{full_dataset}.feature_store",
            job_config=job_config,
        ).result()
        logger.info("Loaded %d feature_store rows", len(all_feature_store))
    logger.info("Refresh BQ complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max number of patients to process")
    args = parser.parse_args()
    run_refresh_bq(limit=args.limit)
