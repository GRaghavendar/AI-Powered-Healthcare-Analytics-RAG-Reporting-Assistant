# Executive Summary

Reporting month: 2025-12

## Key Signals
- Trauma volume was 195 cases, a +28 case change from 2025-11.
- Fatality rate was 3.08% (-1.1 percentage points month over month).
- ICU utilization was 26.67% and ventilator utilization was 11.28%.
- Highest synthetic case volume was at Harborview Emergency Hospital with 996 cases.
- Claims paid amount for the month was $3,650,719.

## Watch List
- Lakeview Regional Medical in 2025-12: Large month-over-month increase (29 cases, z=0.95).
- Central County Hospital in 2025-12: Large month-over-month increase (12 cases, z=0.33).
- Harborview Emergency Hospital in 2025-11: Low volume versus rolling baseline (29 cases, z=-2.53).
- Hudson Valley Trauma Institute in 2025-08: Low volume versus rolling baseline (25 cases, z=-3.28).
- South Bay Trauma Center in 2025-07: Large month-over-month increase (31 cases, z=0.29).

## Forecast
- 2026-01: 197 expected cases (range 155-238).
- 2026-02: 201 expected cases (range 159-243).
- 2026-03: 205 expected cases (range 163-247).

## Data Quality
- Validation checks identified 0 affected rows across synthetic datasets.
- Critical checks include required columns, duplicate keys, direct-PHI column scans, orphan visit IDs, and monthly facility submission gaps.

## Governance Note
- This project uses synthetic data only. The reporting assistant consumes aggregate metrics and de-identified summaries, not patient-level PHI.