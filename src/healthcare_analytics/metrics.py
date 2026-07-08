"""Metric engineering for dashboard-ready public health analytics tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def _pct(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value) * 100, 2)


def _rate(numerator: pd.Series) -> float:
    if len(numerator) == 0:
        return 0.0
    return _pct(numerator.mean())


def executive_overview(data: Dict[str, pd.DataFrame], validation_issues: pd.DataFrame) -> pd.DataFrame:
    trauma = data["trauma_registry"].copy()
    claims = data["claims_data"].copy()
    donor = data["organ_donor_registry"].copy()
    trauma["month"] = _month(trauma["event_month"])
    claims["month"] = _month(claims["service_month"])
    donor["month"] = _month(donor["registration_month"])

    trauma_monthly = (
        trauma.groupby("month")
        .agg(
            total_cases=("visit_id", "count"),
            adult_cases=("adult_pediatric", lambda s: int((s == "Adult").sum())),
            pediatric_cases=("adult_pediatric", lambda s: int((s == "Pediatric").sum())),
            fatality_rate=("fatality_flag", _rate),
            icu_rate=("icu_flag", _rate),
            ventilator_rate=("ventilator_flag", _rate),
            avg_hospital_los_days=("hospital_los_days", "mean"),
            avg_ed_minutes=("ed_minutes", "mean"),
        )
        .reset_index()
    )
    trauma_monthly["avg_hospital_los_days"] = trauma_monthly["avg_hospital_los_days"].round(2)
    trauma_monthly["avg_ed_minutes"] = trauma_monthly["avg_ed_minutes"].round(1)

    claims_monthly = (
        claims.groupby("month")
        .agg(claim_count=("claim_id", "count"), total_paid_amount=("paid_amount", "sum"))
        .reset_index()
    )
    donor_monthly = (
        donor.assign(opt_in_flag=donor["opt_status"].eq("Opt In"))
        .groupby("month")
        .agg(donor_records=("donor_record_id", "count"), donor_opt_in_rate=("opt_in_flag", _rate))
        .reset_index()
    )

    issue_count = 0 if validation_issues.empty else int(validation_issues["rows_affected"].sum())
    overview = trauma_monthly.merge(claims_monthly, on="month", how="left").merge(donor_monthly, on="month", how="left")
    overview["total_paid_amount"] = overview["total_paid_amount"].fillna(0).round(2)
    overview["claim_count"] = overview["claim_count"].fillna(0).astype(int)
    overview["donor_records"] = overview["donor_records"].fillna(0).astype(int)
    overview["donor_opt_in_rate"] = overview["donor_opt_in_rate"].fillna(0)
    overview["data_quality_rows_affected"] = issue_count
    overview["month"] = overview["month"].dt.strftime("%Y-%m")
    return overview.sort_values("month").reset_index(drop=True)


def hospital_comparison(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trauma = data["trauma_registry"].copy()
    hospitals = data["hospital_master"].copy()
    comparison = (
        trauma.groupby("hospital_id")
        .agg(
            total_cases=("visit_id", "count"),
            fatality_rate=("fatality_flag", _rate),
            icu_rate=("icu_flag", _rate),
            ventilator_rate=("ventilator_flag", _rate),
            avg_hospital_los_days=("hospital_los_days", "mean"),
            avg_ed_minutes=("ed_minutes", "mean"),
        )
        .reset_index()
    )
    comparison = hospitals.merge(comparison, on="hospital_id", how="left")
    numeric_columns = ["total_cases", "fatality_rate", "icu_rate", "ventilator_rate", "avg_hospital_los_days", "avg_ed_minutes"]
    comparison[numeric_columns] = comparison[numeric_columns].fillna(0)
    comparison["avg_hospital_los_days"] = comparison["avg_hospital_los_days"].round(2)
    comparison["avg_ed_minutes"] = comparison["avg_ed_minutes"].round(1)
    return comparison.sort_values(["region", "hospital_name"]).reset_index(drop=True)


def demographic_analytics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trauma = data["trauma_registry"].copy()
    rows = []
    for column in ["region", "age_group", "adult_pediatric", "race", "gender"]:
        grouped = (
            trauma.groupby(column)
            .agg(
                total_cases=("visit_id", "count"),
                fatality_rate=("fatality_flag", _rate),
                icu_rate=("icu_flag", _rate),
                avg_hospital_los_days=("hospital_los_days", "mean"),
            )
            .reset_index()
        )
        grouped["segment_type"] = column
        grouped["segment_value"] = grouped[column]
        rows.append(grouped.drop(columns=[column]))
    result = pd.concat(rows, ignore_index=True)
    result["case_share"] = (result["total_cases"] / result["total_cases"].sum() * 100).round(2)
    result["avg_hospital_los_days"] = result["avg_hospital_los_days"].round(2)
    return result[["segment_type", "segment_value", "total_cases", "case_share", "fatality_rate", "icu_rate", "avg_hospital_los_days"]]


def facility_monthly_submissions(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trauma = data["trauma_registry"].copy()
    hospitals = data["hospital_master"].copy()
    trauma["month"] = _month(trauma["event_month"])
    grouped = (
        trauma.groupby(["hospital_id", "month"])
        .agg(
            total_cases=("visit_id", "count"),
            fatality_rate=("fatality_flag", _rate),
            icu_rate=("icu_flag", _rate),
            ventilator_rate=("ventilator_flag", _rate),
            avg_hospital_los_days=("hospital_los_days", "mean"),
        )
        .reset_index()
    )
    grouped = hospitals[["hospital_id", "hospital_name", "region", "trauma_level"]].merge(grouped, on="hospital_id", how="right")
    grouped["avg_hospital_los_days"] = grouped["avg_hospital_los_days"].round(2)
    grouped["month"] = grouped["month"].dt.strftime("%Y-%m")
    return grouped.sort_values(["hospital_id", "month"]).reset_index(drop=True)


def icu_ventilator_metrics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trauma = data["trauma_registry"].copy()
    trauma["month"] = _month(trauma["event_month"])
    result = (
        trauma.groupby(["month", "hospital_id"])
        .agg(
            total_cases=("visit_id", "count"),
            icu_cases=("icu_flag", "sum"),
            ventilator_cases=("ventilator_flag", "sum"),
            avg_icu_los_days=("icu_los_days", "mean"),
            avg_ventilator_days=("ventilator_days", "mean"),
        )
        .reset_index()
    )
    result["icu_rate"] = (result["icu_cases"] / result["total_cases"] * 100).round(2)
    result["ventilator_rate"] = (result["ventilator_cases"] / result["total_cases"] * 100).round(2)
    result["avg_icu_los_days"] = result["avg_icu_los_days"].round(2)
    result["avg_ventilator_days"] = result["avg_ventilator_days"].round(2)
    result["month"] = result["month"].dt.strftime("%Y-%m")
    return result


def ed_ems_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trauma = data["trauma_registry"][["visit_id", "event_month", "hospital_id", "region", "ed_minutes"]].copy()
    ems = data["ems_records"].copy()
    merged = trauma.merge(
        ems[["visit_id", "response_minutes", "scene_minutes", "transport_minutes", "total_pre_hospital_minutes", "recorded_ems_values_flag"]],
        on="visit_id",
        how="left",
    )
    merged["month"] = _month(merged["event_month"])
    result = (
        merged.groupby(["month", "region"])
        .agg(
            trauma_cases=("visit_id", "count"),
            avg_ed_minutes=("ed_minutes", "mean"),
            avg_response_minutes=("response_minutes", "mean"),
            avg_total_pre_hospital_minutes=("total_pre_hospital_minutes", "mean"),
            ems_recorded_rate=("recorded_ems_values_flag", lambda s: _pct(s.fillna(False).mean())),
        )
        .reset_index()
    )
    for column in ["avg_ed_minutes", "avg_response_minutes", "avg_total_pre_hospital_minutes"]:
        result[column] = result[column].round(1)
    result["month"] = result["month"].dt.strftime("%Y-%m")
    return result


def claims_utilization(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    claims = data["claims_data"].copy()
    claims["month"] = _month(claims["service_month"])
    result = (
        claims.groupby(["month", "claim_type", "payer_category"])
        .agg(
            claim_count=("claim_id", "count"),
            allowed_amount=("allowed_amount", "sum"),
            paid_amount=("paid_amount", "sum"),
            avg_paid_amount=("paid_amount", "mean"),
        )
        .reset_index()
    )
    for column in ["allowed_amount", "paid_amount", "avg_paid_amount"]:
        result[column] = result[column].round(2)
    result["month"] = result["month"].dt.strftime("%Y-%m")
    return result


def organ_donor_trends(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    donor = data["organ_donor_registry"].copy()
    donor["month"] = _month(donor["registration_month"])
    grouped = (
        donor.groupby(["month", "region", "opt_status"])
        .agg(records=("donor_record_id", "count"))
        .reset_index()
    )
    pivot = grouped.pivot_table(index=["month", "region"], columns="opt_status", values="records", fill_value=0).reset_index()
    for column in ["Opt In", "Opt Out", "Unknown"]:
        if column not in pivot.columns:
            pivot[column] = 0
    pivot["total_records"] = pivot[["Opt In", "Opt Out", "Unknown"]].sum(axis=1)
    pivot["opt_in_rate"] = np.where(pivot["total_records"] > 0, (pivot["Opt In"] / pivot["total_records"] * 100).round(2), 0)
    pivot["opt_out_rate"] = np.where(pivot["total_records"] > 0, (pivot["Opt Out"] / pivot["total_records"] * 100).round(2), 0)
    pivot["month"] = pivot["month"].dt.strftime("%Y-%m")
    return pivot.rename(columns={"Opt In": "opt_in", "Opt Out": "opt_out", "Unknown": "unknown"})


def data_quality_summary(validation_issues: pd.DataFrame) -> pd.DataFrame:
    if validation_issues.empty:
        return pd.DataFrame(
            [{"dataset": "all", "severity": "none", "issue_count": 0, "rows_affected": 0}]
        )
    return (
        validation_issues.groupby(["dataset", "severity"])
        .agg(issue_count=("check_name", "count"), rows_affected=("rows_affected", "sum"))
        .reset_index()
        .sort_values(["dataset", "severity"])
    )


def build_metric_outputs(data: Dict[str, pd.DataFrame], validation_issues: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "executive_overview": executive_overview(data, validation_issues),
        "hospital_comparison": hospital_comparison(data),
        "demographic_analytics": demographic_analytics(data),
        "facility_monthly_submissions": facility_monthly_submissions(data),
        "icu_ventilator_metrics": icu_ventilator_metrics(data),
        "ed_ems_summary": ed_ems_summary(data),
        "claims_utilization": claims_utilization(data),
        "organ_donor_trends": organ_donor_trends(data),
        "data_quality_issues": validation_issues,
        "data_quality_summary": data_quality_summary(validation_issues),
    }


def save_metric_outputs(outputs: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)
