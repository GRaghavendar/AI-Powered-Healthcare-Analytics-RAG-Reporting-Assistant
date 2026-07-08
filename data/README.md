# Data Directory

This project uses synthetic data only. The pipeline can regenerate all CSV files:

```bash
healthcare-pipeline --records 5000 --months 24 --seed 42
```

Core generated raw files used by the current dashboard:

- `raw/trauma_registry.csv`
- `raw/ems_records.csv`
- `raw/hospital_transfer.csv`
- `raw/claims_data.csv`
- `raw/hospital_master.csv`
- `raw/population_reference.csv`

Generated analytics tables are written to `processed/`.

The generator can create auxiliary synthetic files for validation coverage, but the Streamlit application focuses on trauma, EMS, claims, hospital comparison, forecasting, and data quality outputs.
