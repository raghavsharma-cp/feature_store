"""
Refresh PostgreSQL feature store tables from MongoDB.

Reads all currently admitted patients from MongoDB (same as feature_store.feature_store)
and populates feature_store, vitals_feature_store, documents_feature_store,
notes_feature_store, and orders_feature_store. Replaces existing data.

Only signed orders (order.signed truthy) with category not pending are added to
orders_feature_store and to feature_store.orders; draft, unsigned, and pending orders are skipped.

Usage (from project root):
  python -m scripts.refresh_data
  # or
  PYTHONPATH=. python scripts/refresh_data.py

Requires: db_uri (MongoDB), postgres_url (or POSTGRES_URL) in env or .env / .env.local
"""

import logging
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load env before importing feature_store / api (they use env)
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
    """Parse date from Mongo (datetime, ISO string, or ms timestamp)."""
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


def _mongo_vital_to_row(vital: dict, patient_id: str, vital_id: str) -> "VitalsFeatureStore":
    from api.db.models import VitalsFeatureStore

    row = VitalsFeatureStore(vital_id=vital_id, patient_id=patient_id)
    row.timestamp = _parse_date(vital.get("timestamp"))
    row.therapy_device = (
        vital.get("daysTherapyDevice")
        if isinstance(vital.get("daysTherapyDevice"), (list, dict))
        else None
    )
    row.vent_airway = (
        vital.get("daysVentAirway")
        if isinstance(vital.get("daysVentAirway"), (list, dict))
        else None
    )
    row.temperature = _to_decimal(vital.get("daysTemperature"))
    row.temperature_unit = vital.get("daysTemperatureUnit")
    row.avpu = vital.get("daysAVPU")
    row.hr = _to_int(vital.get("daysHR"))
    row.rr = _to_int(vital.get("daysRR"))
    row.bp = str(vital.get("daysBP")) if vital.get("daysBP") is not None else None
    row.reason_bp = vital.get("daysReasonBP")
    row.map_ = _to_decimal(vital.get("daysMAP"))
    row.cvp = _to_decimal(vital.get("daysCVP"))
    row.sp_o2 = _to_int(vital.get("daysSpO2"))
    row.fi_o2 = _to_int(vital.get("daysFiO2"))
    row.pat_position = vital.get("daysPatPosition")
    row.is_verified = vital.get("isVerified")
    row.verified_time = _parse_date(vital.get("verifiedTime"))
    row.verified_by_name = vital.get("verifiedByName")
    row.verified_by_email = vital.get("verifiedByEmail")
    # abnormal_fields: only store column names that exist in the table; skip unknown types
    _abnormal_type_to_col = {
        "HR": "hr", "RR": "rr", "BP": "bp", "MAP": "map_", "CVP": "cvp",
        "SPO2": "sp_o2", "FIO2": "fi_o2", "TEMPERATURE": "temperature",
        "TEMP": "temperature", "AVPU": "avpu", "PATPOSITION": "pat_position",
        "POSITION": "pat_position",
    }
    _valid_abnormal_cols = set(_abnormal_type_to_col.values())
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
        row.abnormal_fields = col_names if col_names else None
    elif isinstance(vital.get("abnormal_fields"), list):
        row.abnormal_fields = [
            str(x) for x in vital["abnormal_fields"]
            if x is not None and str(x) in _valid_abnormal_cols
        ] or None
    else:
        row.abnormal_fields = None
    return row


def _mongo_document_to_row(doc: dict, patient_id: str, document_id: str) -> "DocumentsFeatureStore":
    from api.db.models import DocumentsFeatureStore

    row = DocumentsFeatureStore(document_id=document_id, patient_id=patient_id)
    row.is_inactive = doc.get("isInactive")
    row.tags = doc.get("tags") if isinstance(doc.get("tags"), (list, dict)) else None
    row.secondary_keys = (
        doc.get("secondaryKeys")
        if isinstance(doc.get("secondaryKeys"), (list, dict))
        else None
    )
    row.snomed_code = (
        doc.get("snomedCode")
        if isinstance(doc.get("snomedCode"), (list, dict))
        else None
    )
    row.key = doc.get("key")
    row.attributes = (
        doc.get("attributes")
        if isinstance(doc.get("attributes"), (list, dict))
        else None
    )
    row.reported_at = _parse_date(doc.get("reportedAt"))
    row.name = doc.get("name")
    row.verified_at = _parse_date(doc.get("verifiedAt"))
    row.classifications = (
        doc.get("classifications")
        if isinstance(doc.get("classifications"), (list, dict))
        else None
    )
    row.label = doc.get("label")
    return row


def _mongo_note_component_to_row(
    note: dict,
    content: dict,
    component: dict,
    patient_id: str,
    note_id: str,
    content_id: str,
    component_id: str,
) -> "NotesFeatureStore":
    """Build one NotesFeatureStore row per (patient_id, note_id, content_id, component_id)."""
    from api.db.models import NotesFeatureStore

    author = content.get("author") or {}
    author_name = author.get("name") if isinstance(author, dict) else None
    author_email = author.get("email") if isinstance(author, dict) else None
    author_role_val = author.get("role") if isinstance(author, dict) else None

    addendum = content.get("addendum")
    if isinstance(addendum, list) and addendum:
        last_add = addendum[-1]
        latest_addendum_author = last_add.get("name") if isinstance(last_add, dict) else None
        latest_addendum_timestamp = _parse_date(last_add.get("timestamp") if isinstance(last_add, dict) else None)
        latest_addendum_text = last_add.get("note") if isinstance(last_add, dict) else None
    else:
        latest_addendum_author = None
        latest_addendum_timestamp = None
        latest_addendum_text = None

    row = NotesFeatureStore(
        patient_id=patient_id,
        note_id=note_id,
        content_id=content_id,
        component_id=component_id,
    )
    row.created_timestamp = _parse_date(
        note.get("createdTimestamp") or note.get("createdAt")
    )
    row.author = author_name
    row.author_id = str(author_email or "")
    row.author_role = author_role_val
    row.author_is_client = content.get("authorIsClient")
    row.note_type = content.get("noteType")
    row.note_sub_type = content.get("noteSubType")
    row.pend_or_signed = content.get("pendOrSigned")
    row.component_raw_value = component.get("value")
    row.component_parsed_text = None
    row.is_delete_enabled = note.get("isDeleteEnabled")
    row.chargeable = str(content.get("chargeable")) if content.get("chargeable") is not None else None
    row.impact_case = content.get("impactCase")
    row.addendum_json = (
        content.get("addendum")
        if isinstance(content.get("addendum"), (list, dict))
        else None
    )
    row.latest_addendum_author = latest_addendum_author
    row.latest_addendum_timestamp = latest_addendum_timestamp
    row.latest_addendum_text = latest_addendum_text
    row.note_diagnoses = []
    row.note_symptoms = []
    row.note_medications = []
    row.note_procedures = []
    row.note_tests = []
    row.api_cost = 0
    return row


def _mongo_order_to_row(order: dict, patient_id: str, order_id: str) -> "OrdersFeatureStore":
    from api.db.models import OrdersFeatureStore

    row = OrdersFeatureStore(order_id=order_id, patient_id=patient_id)
    row.category = order.get("category")
    row.type_ = order.get("type")
    row.state = order.get("state")
    row.created_at = _parse_date(order.get("createdAt"))
    row.updated_at = _parse_date(order.get("updatedAt"))
    row.signed = order.get("signed")
    row.preset = order.get("preset")
    row.orderable = order.get("orderable")
    row.protocol = order.get("protocol")
    row.history = (
        order.get("history")
        if isinstance(order.get("history"), (list, dict))
        else None
    )
    row.additional_information = order.get("additionalInformation")
    row.sos = order.get("sos")
    row.sos_reason = order.get("sosReason")
    row.discontinue_reasons = (
        order.get("discontinueReasons")
        if isinstance(order.get("discontinueReasons"), (list, dict))
        else None
    )
    row.discontinue_at = _parse_date(order.get("discontinueAt"))
    row.discontinue_by = order.get("discontinueBy")
    row.completed_at = _parse_date(order.get("completedAt"))
    row.completed_by = order.get("completedBy")
    row.urgency = order.get("urgency")
    row.start_time = _parse_date(order.get("startTime"))
    row.frequency = (
        order.get("frequency")
        if isinstance(order.get("frequency"), (list, dict))
        else None
    )
    row.number_of_doses = _to_int(order.get("numberOfDoses"))
    row.medication_pta = order.get("medication_pta")
    row.medication_concentration = (
        order.get("medication_concentration")
        if isinstance(order.get("medication_concentration"), (list, dict))
        else None
    )
    row.medication_combination = (
        order.get("medication_combination")
        if isinstance(order.get("medication_combination"), (list, dict))
        else None
    )
    row.medication_name = order.get("medication_name") or order.get("name")
    row.medication_quantity = _to_int(
        order.get("medication_quantity") or order.get("quantity")
    )
    row.medication_unit = order.get("medication_unit") or order.get("unit")
    row.medication_route = order.get("medication_route") or order.get("route")
    row.medication_no_of_days = _to_int(
        order.get("medication_noOfDays") or order.get("noOfDays")
    )
    row.medication_form = order.get("medication_form") or order.get("form")
    row.medication_max_dose = _to_int(
        order.get("medication_maxDose") or order.get("maxDose")
    )
    row.medication_body_weight = _to_int(
        order.get("medication_bodyWeight") or order.get("bodyWeight")
    )
    row.labs_investigation = order.get("labs_investigation") or order.get(
        "investigation"
    )
    row.labs_discipline = order.get("labs_discipline") or order.get("discipline")
    row.labs_specimen_type = order.get("labs_specimenType") or order.get(
        "specimenType"
    )
    row.diets_name = order.get("diets_name") or order.get("name")
    row.diets_rate_value = _to_decimal(
        order.get("diets_rate_value") or order.get("rate", {}).get("value")
    )
    row.diets_rate_unit = order.get("diets_rate_unit") or order.get("rate", {}).get(
        "unit"
    )
    row.communications_gcs = order.get("communications_gcs") or order.get("gcs")
    row.communications_title = order.get("communications_title") or order.get(
        "title"
    )
    row.bloods_start_now = order.get("bloods_startNow") or order.get("startNow")
    row.bloods_time_period_detail = (
        order.get("bloods_timePeriodDetail")
        if isinstance(order.get("bloods_timePeriodDetail"), (list, dict))
        else None
    )
    row.bloods_skip_schedule = (
        order.get("bloods_skipSchedule")
        if isinstance(order.get("bloods_skipSchedule"), (list, dict))
        else None
    )
    row.bloods_title = order.get("bloods_title")
    row.bloods_quantity = _to_int(order.get("bloods_quantity"))
    row.bloods_quantity_unit = order.get("bloods_quantityUnit")
    row.bloods_schedule_selector = order.get("bloods_scheduleSelector")
    row.bloods_snomed_code = _to_int(order.get("bloods_snomedCode"))
    row.procedure_name = order.get("procedure_name")
    row.procedure_ptype = order.get("procedure_ptype")
    row.procedure_laterality = order.get("procedure_laterality")
    row.procedure_site = order.get("procedure_site")
    row.order_no = order.get("orderNo")
    row.lab_id = order.get("labId")
    return row


def _extract_orders_flat(orders_field: Any) -> List[tuple]:
    """Flatten orders (active/pending/completed) into list of (order_id, order_dict)."""
    out: List[tuple] = []
    if not isinstance(orders_field, dict):
        return out
    for bucket in ("active", "pending", "completed"):
        bucket_data = orders_field.get(bucket)
        if not isinstance(bucket_data, dict):
            continue
        for key in (
            "medications",
            "labs",
            "diets",
            "bloods",
            "procedures",
            "vents",
            "communications",
        ):
            items = bucket_data.get(key)
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                oid = item.get("_id") or item.get("id") or f"{bucket}_{key}_{i}"
                out.append((str(oid), item))
    return out


def run_refresh() -> None:
    from api.db.models import (
        DocumentsFeatureStore,
        FeatureStore,
        NotesFeatureStore,
        OrdersFeatureStore,
        VitalsFeatureStore,
    )
    from api.db.repositories.patients import mongo_patient_to_row
    from api.db.session import SessionLocal
    from feature_store.feature_store import BaseFeatureStore, convert_to_serializable

    if not os.environ.get("db_uri"):
        logger.error("db_uri environment variable is not set")
        raise SystemExit(1)

    logger.info("Fetching currently admitted patients from MongoDB")
    store = BaseFeatureStore()
    # Limit fetch to 100 patients to save memory and time; use limit=None to process all
    patients = store.get_all_currently_admitted_patients(limit=100)
    logger.info(f"Processing {len(patients)} patients")

    db = SessionLocal()
    try:
        # Clear existing data (order: child tables first if there were FKs; we have none, so any order)
        for model, name in [
            (VitalsFeatureStore, "vitals_feature_store"),
            (DocumentsFeatureStore, "documents_feature_store"),
            (NotesFeatureStore, "notes_feature_store"),
            (OrdersFeatureStore, "orders_feature_store"),
            (FeatureStore, "feature_store"),
        ]:
            deleted = db.query(model).delete()
            logger.info(f"Cleared {name}: {deleted} rows")
        db.commit()

        for pat in patients:
            serializable = convert_to_serializable(pat)
            if not isinstance(serializable, dict):
                continue
            patient_id = str(serializable.get("_id", ""))

            # feature_store row
            row = mongo_patient_to_row(serializable)
            db.add(row)

            # vitals
            vitals = serializable.get("vitals")
            if isinstance(vitals, list):
                for idx, v in enumerate(vitals):
                    if not isinstance(v, dict):
                        continue
                    vid = str(v.get("_id") or f"{patient_id}_vital_{idx}")
                    v_row = _mongo_vital_to_row(v, patient_id, vid)
                    db.add(v_row)

            # documents
            docs = serializable.get("documents")
            if isinstance(docs, list):
                for idx, d in enumerate(docs):
                    if not isinstance(d, dict):
                        continue
                    doc_id = str(d.get("_id") or d.get("id") or f"{patient_id}_doc_{idx}")
                    d_row = _mongo_document_to_row(d, patient_id, doc_id)
                    db.add(d_row)

            # notes: one row per (patient_id, note_id, content_id, component_id)
            notes_data = serializable.get("notes")
            if isinstance(notes_data, dict):
                final_notes = notes_data.get("finalNotes")
                if isinstance(final_notes, list):
                    for idx, n in enumerate(final_notes):
                        if not isinstance(n, dict):
                            continue
                        note_id = str(
                            n.get("_id") or n.get("id") or f"{patient_id}_note_{idx}"
                        )
                        contents = n.get("content") or []
                        for cidx, content in enumerate(contents):
                            if not isinstance(content, dict):
                                continue
                            content_id = str(content.get("_id") or content.get("id") or f"{note_id}_c_{cidx}")
                            components = content.get("components") or []
                            for comp_idx, component in enumerate(components):
                                if not isinstance(component, dict):
                                    continue
                                comp_id = str(component.get("id") or f"_{comp_idx}")
                                n_row = _mongo_note_component_to_row(
                                    n, content, component,
                                    patient_id, note_id, content_id, comp_id,
                                )
                                db.add(n_row)
            elif isinstance(notes_data, list):
                for idx, n in enumerate(notes_data):
                    if not isinstance(n, dict):
                        continue
                    note_id = str(
                        n.get("_id") or n.get("id") or f"{patient_id}_note_{idx}"
                    )
                    contents = n.get("content") or []
                    for cidx, content in enumerate(contents):
                        if not isinstance(content, dict):
                            continue
                        content_id = str(content.get("_id") or content.get("id") or f"{note_id}_c_{cidx}")
                        components = content.get("components") or []
                        for comp_idx, component in enumerate(components):
                            if not isinstance(component, dict):
                                continue
                            comp_id = str(component.get("id") or f"_{comp_idx}")
                            n_row = _mongo_note_component_to_row(
                                n, content, component,
                                patient_id, note_id, content_id, comp_id,
                            )
                            db.add(n_row)

            # orders: only signed orders (order.signed truthy) and category not pending are added
            orders_data = serializable.get("orders")
            signed_order_ids: List[str] = []
            for oid, o in _extract_orders_flat(orders_data):
                if not o.get("signed") or (
                    o.get("category") and str(o.get("category")).strip().lower() == "pending"
                ):
                    continue
                signed_order_ids.append(oid)
                o_row = _mongo_order_to_row(o, patient_id, oid)
                db.add(o_row)
            row.orders = signed_order_ids or None

        db.commit()
        logger.info("Committed all feature store tables")
    finally:
        db.close()

    logger.info("Refresh complete")


if __name__ == "__main__":
    run_refresh()
