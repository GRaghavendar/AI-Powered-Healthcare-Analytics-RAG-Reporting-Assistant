"""Synthetic data generation for public health trauma analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import (
    AGE_GROUPS,
    CLAIM_TYPES,
    DEFAULT_END_MONTH,
    DEFAULT_MONTHS,
    DEFAULT_RECORDS,
    GENDER_CATEGORIES,
    HOSPITAL_RECORDS,
    INJURY_TYPES,
    PAYER_CATEGORIES,
    RACE_CATEGORIES,
    REGION_POPULATION,
    REGISTRATION_LOCATIONS,
)


def month_sequence(months: int = DEFAULT_MONTHS, end_month: str = DEFAULT_END_MONTH) -> pd.DatetimeIndex:
    """Return a stable sequence of month starts."""

    end = pd.Timestamp(end_month).to_period("M").to_timestamp()
    return pd.date_range(end=end, periods=months, freq="MS")


def hospital_master() -> pd.DataFrame:
    return pd.DataFrame(HOSPITAL_RECORDS)


def population_reference() -> pd.DataFrame:
    rows = []
    for region, population in REGION_POPULATION.items():
        rows.append(
            {
                "region": region,
                "population": population,
                "adult_population": int(population * 0.79),
                "pediatric_population": int(population * 0.21),
            }
        )
    return pd.DataFrame(rows)


def _age_group(age: int) -> str:
    if age <= 17:
        return "0-17"
    if age <= 34:
        return "18-34"
    if age <= 49:
        return "35-49"
    if age <= 64:
        return "50-64"
    return "65+"


def _choice(rng: np.random.Generator, values: list[str], probs: list[float] | None = None, size=None):
    return rng.choice(values, p=probs, size=size)


def generate_trauma_registry(
    records: int = DEFAULT_RECORDS,
    months: int = DEFAULT_MONTHS,
    seed: int = 42,
    end_month: str = DEFAULT_END_MONTH,
) -> pd.DataFrame:
    """Generate synthetic trauma visit records.

    The data is intentionally aggregate-friendly and contains no names, dates of
    birth, addresses, medical record numbers, or other direct identifiers.
    """

    rng = np.random.default_rng(seed)
    hospitals = hospital_master()
    month_values = month_sequence(months, end_month)
    seasonality = np.array([1.0 + 0.16 * np.sin((m.month - 1) / 12 * 2 * np.pi) for m in month_values])
    month_prob = seasonality / seasonality.sum()

    hospital_weights = hospitals["bed_count"].to_numpy(dtype=float)
    hospital_weights = hospital_weights / hospital_weights.sum()
    hospital_ids = _choice(rng, hospitals["hospital_id"].tolist(), hospital_weights.tolist(), records)
    hospital_lookup = hospitals.set_index("hospital_id")

    ages = np.where(
        rng.random(records) < 0.18,
        rng.integers(0, 18, size=records),
        rng.integers(18, 91, size=records),
    )
    severity = np.clip(rng.gamma(shape=2.0, scale=2.0, size=records), 1, 10).round(1)
    icu_prob = np.clip(0.04 + severity * 0.055 + (ages > 70) * 0.05, 0.02, 0.72)
    icu_flag = rng.random(records) < icu_prob
    ventilator_prob = np.where(icu_flag, np.clip(0.10 + severity * 0.045, 0.05, 0.55), 0.025)
    ventilator_flag = rng.random(records) < ventilator_prob
    fatality_prob = np.clip(0.006 + severity * 0.009 + icu_flag * 0.035 + ventilator_flag * 0.055, 0.002, 0.45)
    fatality_flag = rng.random(records) < fatality_prob

    hospital_los = np.maximum(
        1,
        rng.poisson(2 + severity * 0.8 + icu_flag * 3 + ventilator_flag * 2, size=records),
    )
    icu_los = np.where(icu_flag, np.minimum(hospital_los, rng.poisson(1 + severity * 0.5, size=records) + 1), 0)
    ventilator_days = np.where(
        ventilator_flag,
        np.maximum(1, np.minimum(icu_los + 1, rng.poisson(1 + severity * 0.35, size=records) + 1)),
        0,
    )
    ed_minutes = np.clip(rng.normal(195 + severity * 12, 55, size=records), 45, 720).round().astype(int)
    event_months = _choice(rng, list(month_values), month_prob.tolist(), records)

    rows = []
    for idx in range(records):
        hospital_id = hospital_ids[idx]
        hospital = hospital_lookup.loc[hospital_id]
        adult_pediatric = "Pediatric" if ages[idx] < 18 else "Adult"
        if adult_pediatric == "Pediatric" and not bool(hospital["pediatric_capable"]):
            hospital_id = _choice(
                rng,
                hospitals.loc[hospitals["pediatric_capable"], "hospital_id"].tolist(),
            )
            hospital = hospital_lookup.loc[hospital_id]

        if fatality_flag[idx]:
            discharge_status = "Expired"
        else:
            discharge_status = _choice(
                rng,
                ["Home", "Rehabilitation", "Skilled Nursing", "Transferred", "Other"],
                [0.49, 0.18, 0.14, 0.13, 0.06],
            )

        rows.append(
            {
                "visit_id": f"TRM{idx + 1:07d}",
                "event_month": pd.Timestamp(event_months[idx]).strftime("%Y-%m-01"),
                "hospital_id": hospital_id,
                "region": hospital["region"],
                "age": int(ages[idx]),
                "age_group": _age_group(int(ages[idx])),
                "adult_pediatric": adult_pediatric,
                "race": _choice(
                    rng,
                    RACE_CATEGORIES,
                    [0.01, 0.08, 0.14, 0.18, 0.53, 0.06],
                ),
                "gender": _choice(rng, GENDER_CATEGORIES, [0.47, 0.51, 0.02]),
                "injury_type": _choice(rng, INJURY_TYPES, [0.36, 0.25, 0.11, 0.05, 0.07, 0.16]),
                "severity_score": float(severity[idx]),
                "fatality_flag": bool(fatality_flag[idx]),
                "discharge_status": discharge_status,
                "icu_flag": bool(icu_flag[idx]),
                "icu_los_days": int(icu_los[idx]),
                "ventilator_flag": bool(ventilator_flag[idx]),
                "ventilator_days": int(ventilator_days[idx]),
                "hospital_los_days": int(hospital_los[idx]),
                "ed_minutes": int(ed_minutes[idx]),
                "ems_arrival_mode": _choice(rng, ["Ground EMS", "Air EMS", "Walk-in", "Transfer"], [0.68, 0.05, 0.17, 0.10]),
            }
        )
    return pd.DataFrame(rows)


def generate_ems_records(trauma: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    ems_visits = trauma.sample(frac=0.84, random_state=seed).copy()
    n = len(ems_visits)
    response = np.clip(rng.normal(10, 4, n), 2, 45).round(1)
    scene = np.clip(rng.normal(18, 7, n), 3, 80).round(1)
    transport = np.clip(rng.normal(24, 10, n), 5, 120).round(1)
    return pd.DataFrame(
        {
            "ems_record_id": [f"EMS{i + 1:07d}" for i in range(n)],
            "visit_id": ems_visits["visit_id"].to_numpy(),
            "event_month": ems_visits["event_month"].to_numpy(),
            "hospital_id": ems_visits["hospital_id"].to_numpy(),
            "region": ems_visits["region"].to_numpy(),
            "response_minutes": response,
            "scene_minutes": scene,
            "transport_minutes": transport,
            "total_pre_hospital_minutes": response + scene + transport,
            "recorded_ems_values_flag": rng.random(n) > 0.04,
            "arrival_mode": ems_visits["ems_arrival_mode"].to_numpy(),
        }
    )


def generate_hospital_transfer(trauma: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    hospitals = hospital_master()
    hospital_ids = hospitals["hospital_id"].tolist()
    transfer_prob = np.clip(0.04 + trauma["severity_score"].to_numpy() * 0.018, 0.02, 0.34)
    transferred = trauma.loc[rng.random(len(trauma)) < transfer_prob].copy()
    rows = []
    for i, row in enumerate(transferred.itertuples(index=False)):
        transfer_hospital = row.hospital_id
        referring = rng.choice([h for h in hospital_ids if h != transfer_hospital])
        final = transfer_hospital if rng.random() > 0.08 else rng.choice(hospital_ids)
        rows.append(
            {
                "transfer_record_id": f"TRF{i + 1:07d}",
                "visit_id": row.visit_id,
                "event_month": row.event_month,
                "referring_hospital_id": referring,
                "transfer_hospital_id": transfer_hospital,
                "final_hospital_id": final,
                "transfer_required_flag": True,
                "transfer_type": rng.choice(["ED to trauma center", "Interfacility", "Specialty care", "Return transfer"]),
            }
        )
    return pd.DataFrame(rows)


def generate_claims_data(trauma: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 23)
    rows = []
    claim_id = 1
    for row in trauma.itertuples(index=False):
        claim_count = 1 + int(rng.random() < 0.20) + int(rng.random() < 0.06)
        for _ in range(claim_count):
            claim_type = rng.choice(CLAIM_TYPES, p=[0.48, 0.20, 0.07, 0.25])
            base = {
                "Inpatient": 18500,
                "Outpatient": 2200,
                "Ambulatory Surgery": 9300,
                "Emergency Department": 4100,
            }[claim_type]
            multiplier = 1 + row.severity_score * 0.13 + row.icu_flag * 0.45 + row.ventilator_flag * 0.55
            allowed = max(250, rng.normal(base * multiplier, base * 0.35))
            paid = allowed * rng.uniform(0.62, 0.93)
            rows.append(
                {
                    "claim_id": f"CLM{claim_id:08d}",
                    "visit_id": row.visit_id,
                    "service_month": row.event_month,
                    "hospital_id": row.hospital_id,
                    "claim_type": claim_type,
                    "payer_category": rng.choice(PAYER_CATEGORIES, p=[0.36, 0.24, 0.27, 0.06, 0.07]),
                    "diagnosis_group": rng.choice(["Orthopedic", "Neuro", "Thoracic", "Abdominal", "Burn", "Other"]),
                    "allowed_amount": round(float(allowed), 2),
                    "paid_amount": round(float(paid), 2),
                }
            )
            claim_id += 1
    return pd.DataFrame(rows)


def generate_organ_donor_registry(
    records: int,
    months: int = DEFAULT_MONTHS,
    seed: int = 42,
    end_month: str = DEFAULT_END_MONTH,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 31)
    month_values = month_sequence(months, end_month)
    regions = list(REGION_POPULATION.keys())
    region_weights = np.array(list(REGION_POPULATION.values()), dtype=float)
    region_weights = region_weights / region_weights.sum()
    rows = []
    for idx in range(max(200, int(records * 0.35))):
        region = rng.choice(regions, p=region_weights)
        opt_status = rng.choice(["Opt In", "Opt Out", "Unknown"], p=[0.68, 0.21, 0.11])
        rows.append(
            {
                "donor_record_id": f"DON{idx + 1:07d}",
                "registration_month": pd.Timestamp(rng.choice(month_values)).strftime("%Y-%m-01"),
                "region": region,
                "age_group": rng.choice(AGE_GROUPS, p=[0.16, 0.22, 0.21, 0.23, 0.18]),
                "race": rng.choice(RACE_CATEGORIES, p=[0.01, 0.08, 0.13, 0.18, 0.54, 0.06]),
                "gender": rng.choice(GENDER_CATEGORIES, p=[0.49, 0.49, 0.02]),
                "registration_location": rng.choice(REGISTRATION_LOCATIONS, p=[0.43, 0.25, 0.11, 0.12, 0.09]),
                "opt_status": opt_status,
                "active_status": rng.choice(["Active", "Deceased", "Inactive"], p=[0.90, 0.03, 0.07]),
                "comments_category": rng.choice(
                    ["No comment", "Needs follow-up", "Duplicate review", "Address update", "Preference clarification"],
                    p=[0.72, 0.07, 0.05, 0.08, 0.08],
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_all(
    output_dir: str | Path,
    records: int = DEFAULT_RECORDS,
    months: int = DEFAULT_MONTHS,
    seed: int = 42,
    end_month: str = DEFAULT_END_MONTH,
) -> Dict[str, pd.DataFrame]:
    """Generate all synthetic datasets and write them to CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trauma = generate_trauma_registry(records=records, months=months, seed=seed, end_month=end_month)
    data = {
        "hospital_master": hospital_master(),
        "population_reference": population_reference(),
        "trauma_registry": trauma,
        "ems_records": generate_ems_records(trauma, seed=seed),
        "hospital_transfer": generate_hospital_transfer(trauma, seed=seed),
        "claims_data": generate_claims_data(trauma, seed=seed),
        "organ_donor_registry": generate_organ_donor_registry(records=records, months=months, seed=seed, end_month=end_month),
    }

    for name, frame in data.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)
    return data
