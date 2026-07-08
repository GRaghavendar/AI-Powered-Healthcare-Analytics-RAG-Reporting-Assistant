"""Reusable data quality checks for synthetic public health datasets."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "trauma_registry": [
        "visit_id",
        "event_month",
        "hospital_id",
        "region",
        "age",
        "age_group",
        "adult_pediatric",
        "race",
        "gender",
        "fatality_flag",
        "icu_flag",
        "ventilator_flag",
        "hospital_los_days",
        "ed_minutes",
    ],
    "ems_records": [
        "ems_record_id",
        "visit_id",
        "event_month",
        "hospital_id",
        "response_minutes",
        "scene_minutes",
        "transport_minutes",
    ],
    "hospital_transfer": [
        "transfer_record_id",
        "visit_id",
        "referring_hospital_id",
        "transfer_hospital_id",
        "final_hospital_id",
    ],
    "claims_data": [
        "claim_id",
        "visit_id",
        "service_month",
        "hospital_id",
        "claim_type",
        "payer_category",
        "allowed_amount",
        "paid_amount",
    ],
    "organ_donor_registry": [
        "donor_record_id",
        "registration_month",
        "region",
        "age_group",
        "race",
        "gender",
        "registration_location",
        "opt_status",
        "active_status",
    ],
}

PRIMARY_KEYS = {
    "trauma_registry": "visit_id",
    "ems_records": "ems_record_id",
    "hospital_transfer": "transfer_record_id",
    "claims_data": "claim_id",
    "organ_donor_registry": "donor_record_id",
}

DIRECT_PHI_KEYWORDS = {
    "first_name",
    "last_name",
    "dob",
    "date_of_birth",
    "ssn",
    "social_security",
    "address",
    "phone",
    "email",
    "medical_record_number",
    "mrn",
}


def _issue(dataset: str, check_name: str, severity: str, rows_affected: int, details: str) -> dict:
    return {
        "dataset": dataset,
        "check_name": check_name,
        "severity": severity,
        "rows_affected": int(rows_affected),
        "details": details,
    }


def _missing_required_columns(dataset: str, frame: pd.DataFrame) -> list[dict]:
    missing = [col for col in REQUIRED_COLUMNS.get(dataset, []) if col not in frame.columns]
    if not missing:
        return []
    return [_issue(dataset, "required_columns", "critical", len(missing), f"Missing columns: {', '.join(missing)}")]


def _duplicate_key_check(dataset: str, frame: pd.DataFrame) -> list[dict]:
    key = PRIMARY_KEYS.get(dataset)
    if not key or key not in frame.columns:
        return []
    duplicate_count = int(frame[key].duplicated().sum())
    if duplicate_count == 0:
        return []
    return [_issue(dataset, "duplicate_primary_key", "high", duplicate_count, f"Duplicate values found in {key}.")]


def _null_check(dataset: str, frame: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    issues = []
    for column in columns:
        if column in frame.columns:
            missing = int(frame[column].isna().sum())
            if missing:
                issues.append(_issue(dataset, f"missing_{column}", "medium", missing, f"{column} contains null values."))
    return issues


def _allowed_values_check(dataset: str, frame: pd.DataFrame, column: str, allowed: Iterable[str]) -> list[dict]:
    if column not in frame.columns:
        return []
    allowed_set = set(allowed)
    invalid = ~frame[column].isin(allowed_set)
    count = int(invalid.sum())
    if count == 0:
        return []
    return [_issue(dataset, f"allowed_values_{column}", "medium", count, f"Unexpected values in {column}.")]


def _non_negative_check(dataset: str, frame: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    issues = []
    for column in columns:
        if column in frame.columns:
            count = int((pd.to_numeric(frame[column], errors="coerce") < 0).sum())
            if count:
                issues.append(_issue(dataset, f"non_negative_{column}", "high", count, f"{column} has negative values."))
    return issues


def _phi_column_check(dataset: str, frame: pd.DataFrame) -> list[dict]:
    lower_columns = {column.lower() for column in frame.columns}
    found = sorted(lower_columns.intersection(DIRECT_PHI_KEYWORDS))
    if not found:
        return []
    return [_issue(dataset, "direct_phi_column_scan", "critical", len(found), f"Potential direct PHI columns found: {', '.join(found)}")]


def validate_trauma_registry(frame: pd.DataFrame) -> list[dict]:
    dataset = "trauma_registry"
    issues = []
    issues.extend(_missing_required_columns(dataset, frame))
    issues.extend(_duplicate_key_check(dataset, frame))
    issues.extend(_phi_column_check(dataset, frame))
    issues.extend(_null_check(dataset, frame, ["visit_id", "event_month", "hospital_id", "age_group", "adult_pediatric"]))
    issues.extend(_allowed_values_check(dataset, frame, "adult_pediatric", ["Adult", "Pediatric"]))
    issues.extend(_allowed_values_check(dataset, frame, "gender", ["Female", "Male", "Unknown"]))
    issues.extend(_non_negative_check(dataset, frame, ["age", "icu_los_days", "ventilator_days", "hospital_los_days", "ed_minutes"]))

    if {"icu_los_days", "hospital_los_days"}.issubset(frame.columns):
        count = int((frame["icu_los_days"] > frame["hospital_los_days"]).sum())
        if count:
            issues.append(_issue(dataset, "icu_los_not_greater_than_hospital_los", "high", count, "ICU days exceed hospital LOS."))
    if {"ventilator_flag", "ventilator_days"}.issubset(frame.columns):
        count = int(((~frame["ventilator_flag"].astype(bool)) & (frame["ventilator_days"] > 0)).sum())
        if count:
            issues.append(_issue(dataset, "ventilator_day_consistency", "medium", count, "Ventilator days exist when ventilator flag is false."))
    return issues


def validate_cross_dataset(data: Dict[str, pd.DataFrame]) -> list[dict]:
    issues = []
    trauma = data.get("trauma_registry", pd.DataFrame())
    if "visit_id" not in trauma.columns:
        return issues

    visit_ids = set(trauma["visit_id"])
    for dataset in ["ems_records", "hospital_transfer", "claims_data"]:
        frame = data.get(dataset, pd.DataFrame())
        if "visit_id" not in frame.columns or frame.empty:
            continue
        orphan_count = int((~frame["visit_id"].isin(visit_ids)).sum())
        if orphan_count:
            issues.append(_issue(dataset, "orphan_visit_id", "high", orphan_count, "Records do not map back to trauma_registry.visit_id."))

    hospitals = data.get("hospital_master", pd.DataFrame())
    if {"hospital_id"}.issubset(trauma.columns) and {"hospital_id"}.issubset(hospitals.columns):
        known_hospitals = set(hospitals["hospital_id"])
        count = int((~trauma["hospital_id"].isin(known_hospitals)).sum())
        if count:
            issues.append(_issue("trauma_registry", "unknown_hospital_id", "critical", count, "Trauma records use hospital IDs outside hospital_master."))

    return issues


def validate_facility_submissions(data: Dict[str, pd.DataFrame]) -> list[dict]:
    trauma = data.get("trauma_registry", pd.DataFrame())
    hospitals = data.get("hospital_master", pd.DataFrame())
    if trauma.empty or hospitals.empty or not {"hospital_id", "event_month"}.issubset(trauma.columns):
        return []

    months = pd.to_datetime(trauma["event_month"]).dt.to_period("M").dt.to_timestamp().sort_values().unique()
    expected = pd.MultiIndex.from_product(
        [hospitals["hospital_id"].unique(), months],
        names=["hospital_id", "event_month"],
    ).to_frame(index=False)
    actual = (
        trauma.assign(event_month=pd.to_datetime(trauma["event_month"]).dt.to_period("M").dt.to_timestamp())
        .groupby(["hospital_id", "event_month"])
        .size()
        .reset_index(name="records")
    )
    merged = expected.merge(actual, on=["hospital_id", "event_month"], how="left")
    missing_count = int(merged["records"].isna().sum())
    if missing_count == 0:
        return []
    return [
        _issue(
            "trauma_registry",
            "facility_monthly_submission_gap",
            "medium",
            missing_count,
            "Hospital-month combinations have zero trauma submissions in the synthetic feed.",
        )
    ]


def run_validation(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run all validation checks and return a tidy issue table."""

    issues: list[dict] = []
    for dataset, frame in data.items():
        if dataset in REQUIRED_COLUMNS:
            issues.extend(_missing_required_columns(dataset, frame))
            issues.extend(_duplicate_key_check(dataset, frame))
            issues.extend(_phi_column_check(dataset, frame))

    if "trauma_registry" in data:
        issues.extend(validate_trauma_registry(data["trauma_registry"]))

    for dataset, frame in data.items():
        if dataset != "trauma_registry" and dataset in REQUIRED_COLUMNS:
            issues.extend(_null_check(dataset, frame, [PRIMARY_KEYS.get(dataset, ""), "event_month", "service_month", "registration_month"]))
            issues.extend(_non_negative_check(dataset, frame, ["allowed_amount", "paid_amount", "response_minutes", "scene_minutes", "transport_minutes"]))

    issues.extend(validate_cross_dataset(data))
    issues.extend(validate_facility_submissions(data))

    columns = ["dataset", "check_name", "severity", "rows_affected", "details"]
    if not issues:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(issues)[columns].drop_duplicates().reset_index(drop=True)
