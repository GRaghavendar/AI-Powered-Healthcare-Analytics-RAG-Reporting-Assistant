# Data Validation Rules

## Purpose

The validation layer runs before metric generation, dashboard refresh, reporting summaries, and RAG indexing. Its purpose is to identify data issues early so analysts can trust the aggregate outputs used by the application.

## Validation Scope

Validation covers the following input domains:

- Hospital master records
- Population reference data
- Trauma registry records
- EMS records
- Hospital transfer records
- Claims records
- Organ donor registration records

## Required Column Checks

Each dataset has a required schema. Missing required columns are marked as critical because downstream metrics may fail or become misleading.

Examples:

- `visit_id` in trauma registry records
- `hospital_id` in facility-level datasets
- `event_month`, `service_month`, or `registration_month` in time-based datasets
- Core category fields used for reporting segments

## Duplicate Key Checks

Primary identifiers are checked for duplicate values.

Expected unique keys:

- `visit_id` for trauma registry records
- `ems_record_id` for EMS records
- `transfer_record_id` for transfer events
- `claim_id` for claims records
- `donor_record_id` for organ donor records

Duplicate keys are flagged because they can inflate counts, duplicate payments, or distort facility-level reporting.

## Missing Value Checks

Important fields are checked for missing values before metrics are calculated.

Common checks include:

- Facility identifiers
- Visit identifiers
- Reporting month fields
- Age group
- Adult or pediatric category
- Claim type
- Donor registration status

## Allowed Value Checks

Categorical fields are checked against expected values.

Examples:

- Adult or pediatric category should be `Adult` or `Pediatric`.
- Gender should be `Female`, `Male`, or `Unknown`.
- Payer category should match the configured payer list.
- Claim type should match the configured claim type list.

## Numeric Range Checks

Numeric duration and amount fields should not be negative.

Examples:

- ICU days
- Ventilator days
- Hospital length of stay
- ED minutes
- EMS response, scene, and transport minutes
- Allowed amount
- Paid amount

## Cross-Dataset Reconciliation

EMS, transfer, and claims rows should map back to a trauma registry `visit_id`. Rows that cannot be reconciled are flagged as orphan records.

Why this matters:

- Orphan EMS rows cannot be connected to the main trauma visit.
- Orphan claims rows can distort utilization totals.
- Orphan transfer rows can affect facility movement analysis.

## Facility Submission Review

The validation layer checks hospital-month combinations to identify possible missing facility submissions or unusual reporting gaps.

Primary uses:

- Support analyst follow-up.
- Identify incomplete reporting periods.
- Protect dashboard users from interpreting missing submissions as true zero activity.

## PHI-Oriented Schema Scan

The application scans column names for direct identifier patterns.

Examples of blocked or flagged fields:

- First name
- Last name
- Date of birth
- Social Security number
- Street address
- Phone number
- Email address
- Medical record number
- MRN

The local sample data generator does not create these fields.

## Severity Levels

Validation results are summarized by severity.

- `critical`: Can stop or materially distort metric generation.
- `warning`: Should be reviewed before publication or stakeholder use.
- `informational`: Useful for monitoring but not blocking.

## Analyst Review

Validation output should be reviewed before using the dashboard for operational reporting. The application summarizes issues, but final interpretation remains an analyst responsibility.
