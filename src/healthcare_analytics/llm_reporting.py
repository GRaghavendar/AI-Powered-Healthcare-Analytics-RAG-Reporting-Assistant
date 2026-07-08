"""Deterministic aggregate reporting summaries used alongside the RAG assistant."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def _delta(current: float, previous: float) -> str:
    change = current - previous
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}"


def _last_two(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    ordered = frame.sort_values("month").reset_index(drop=True)
    last = ordered.iloc[-1]
    previous = ordered.iloc[-2] if len(ordered) > 1 else None
    return last, previous


def generate_executive_summary(
    metric_outputs: Dict[str, pd.DataFrame],
    anomalies: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
) -> str:
    """Create a stakeholder-ready narrative from aggregate metrics."""

    overview = metric_outputs["executive_overview"]
    comparison = metric_outputs["hospital_comparison"]
    dq_summary = metric_outputs["data_quality_summary"]
    last, previous = _last_two(overview)

    lines = [
        "# Executive Summary",
        "",
        f"Reporting month: {last['month']}",
        "",
        "## Key Signals",
    ]

    if previous is not None:
        case_delta = int(last["total_cases"] - previous["total_cases"])
        lines.append(
            f"- Trauma volume was {int(last['total_cases']):,} cases, a {case_delta:+,} case change from {previous['month']}."
        )
        lines.append(
            f"- Fatality rate was {last['fatality_rate']:.2f}% ({_delta(last['fatality_rate'], previous['fatality_rate'])} percentage points month over month)."
        )
        lines.append(
            f"- ICU utilization was {last['icu_rate']:.2f}% and ventilator utilization was {last['ventilator_rate']:.2f}%."
        )
    else:
        lines.append(f"- Trauma volume was {int(last['total_cases']):,} cases.")
        lines.append(f"- Fatality rate was {last['fatality_rate']:.2f}%.")

    top_hospital = comparison.sort_values("total_cases", ascending=False).iloc[0]
    lines.append(
        f"- Highest synthetic case volume was at {top_hospital['hospital_name']} with {int(top_hospital['total_cases']):,} cases."
    )
    lines.append(f"- Claims paid amount for the month was ${last['total_paid_amount']:,.0f}.")

    lines.extend(["", "## Watch List"])
    if anomalies is not None and not anomalies.empty:
        latest_anomalies = anomalies.loc[anomalies["anomaly_flag"]].sort_values("month", ascending=False).head(5)
        if latest_anomalies.empty:
            lines.append("- No facility-month submission anomalies were flagged by the rolling baseline detector.")
        else:
            for row in latest_anomalies.itertuples(index=False):
                lines.append(
                    f"- {row.hospital_name} in {row.month}: {row.anomaly_reason} ({int(row.total_cases)} cases, z={row.z_score})."
                )
    else:
        lines.append("- No anomaly table was available.")

    if forecasts is not None and not forecasts.empty:
        lines.extend(["", "## Forecast"])
        for row in forecasts.itertuples(index=False):
            lines.append(
                f"- {row.month}: {row.forecast_total_cases:,} expected cases "
                f"(range {row.lower_bound:,}-{row.upper_bound:,})."
            )

    issue_rows = int(dq_summary["rows_affected"].sum()) if not dq_summary.empty else 0
    lines.extend(
        [
            "",
            "## Data Quality",
            f"- Validation checks identified {issue_rows:,} affected rows across synthetic datasets.",
            "- Critical checks include required columns, duplicate keys, direct-PHI column scans, orphan visit IDs, and monthly facility submission gaps.",
            "",
            "## Governance Note",
            "- This project uses synthetic data only. The reporting assistant consumes aggregate metrics and de-identified summaries, not patient-level PHI.",
        ]
    )
    return "\n".join(lines)


def generate_facility_narratives(hospital_comparison: pd.DataFrame, anomalies: pd.DataFrame | None = None) -> str:
    lines = ["# Facility Narratives", ""]
    anomaly_lookup = {}
    if anomalies is not None and not anomalies.empty:
        flagged = anomalies.loc[anomalies["anomaly_flag"]]
        anomaly_lookup = flagged.groupby("hospital_id")["anomaly_reason"].apply(lambda s: "; ".join(sorted(set(s)))).to_dict()

    for row in hospital_comparison.sort_values("total_cases", ascending=False).itertuples(index=False):
        watch_note = anomaly_lookup.get(row.hospital_id, "No recent anomaly flags")
        lines.append(f"## {row.hospital_name}")
        lines.append(
            f"{row.hospital_name} reported {int(row.total_cases):,} synthetic trauma cases. "
            f"Fatality rate was {row.fatality_rate:.2f}%, ICU utilization was {row.icu_rate:.2f}%, "
            f"and average hospital length of stay was {row.avg_hospital_los_days:.2f} days. "
            f"Watch list: {watch_note}."
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
