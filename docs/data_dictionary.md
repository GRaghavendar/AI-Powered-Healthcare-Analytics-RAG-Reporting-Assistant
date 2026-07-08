# Data Dictionary

All files use non-identifiable sample data for local analytics, testing, and documentation.

The current dashboard focuses on trauma registry, EMS, claims, hospital reference, population, forecasting, and data quality outputs. Auxiliary synthetic files may exist in the generated raw folder for validation coverage.

## Raw Inputs

| File | Grain | Key Fields |
|---|---|---|
| `trauma_registry.csv` | One row per synthetic trauma visit | `visit_id`, `event_month`, `hospital_id`, `age_group`, `adult_pediatric`, `race`, `gender`, `fatality_flag`, `icu_flag`, `ventilator_flag`, `hospital_los_days`, `ed_minutes` |
| `ems_records.csv` | One row per EMS record | `ems_record_id`, `visit_id`, `response_minutes`, `scene_minutes`, `transport_minutes`, `recorded_ems_values_flag` |
| `hospital_transfer.csv` | One row per transfer event | `transfer_record_id`, `visit_id`, `referring_hospital_id`, `transfer_hospital_id`, `final_hospital_id` |
| `claims_data.csv` | One row per synthetic claim line | `claim_id`, `visit_id`, `claim_type`, `payer_category`, `allowed_amount`, `paid_amount` |
| `hospital_master.csv` | One row per synthetic hospital | `hospital_id`, `hospital_name`, `region`, `trauma_level`, `bed_count`, `pediatric_capable` |
| `population_reference.csv` | One row per region | `region`, `population`, `adult_population`, `pediatric_population` |

## Processed Outputs

| File | Purpose |
|---|---|
| `executive_overview.csv` | Monthly public health KPIs and claims rollups. |
| `hospital_comparison.csv` | Trauma center comparison table. |
| `demographic_analytics.csv` | Segment-level trauma metrics by region, age group, race, gender, and adult/pediatric flag. |
| `facility_monthly_submissions.csv` | Hospital-month submission and KPI table. |
| `icu_ventilator_metrics.csv` | ICU and ventilator utilization metrics. |
| `ed_ems_summary.csv` | ED and EMS timing metrics. |
| `claims_utilization.csv` | Claims volume and payment metrics by claim and payer type. |
| `data_quality_issues.csv` | Detailed validation issues. |
| `data_quality_summary.csv` | Validation issue summary. |
| `reporting_anomalies.csv` | Rolling baseline anomaly flags. |
| `case_volume_forecast.csv` | Future monthly case volume forecast. |
