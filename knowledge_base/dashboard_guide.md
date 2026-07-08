# Dashboard Guide

## Purpose

The dashboard provides a consolidated view of trauma reporting, data quality, utilization, and supporting documentation. It is designed for analysts who need to review aggregate trends, compare facilities, validate reporting completeness, and answer operational questions.

## Executive Overview

The executive overview summarizes the latest reporting period and recent trend lines.

Key measures:

- Total trauma cases
- Adult cases
- Pediatric cases
- Fatality rate
- ICU utilization rate
- Ventilator utilization rate
- Claims paid amount
- Data quality impact

Primary use:

Quickly understand current reporting volume, high-level outcomes, and operational signals.

## Trauma Center Comparison

This view compares hospitals and trauma centers using aggregate measures.

Key measures:

- Total cases
- Fatality rate
- ICU rate
- Ventilator rate
- Average hospital length of stay
- Average ED minutes

Primary use:

Identify differences in case volume, care intensity, and reporting patterns across facilities.

## Adult Vs Pediatric

This view separates trauma cases into adult and pediatric categories.

Primary use:

Support service planning, pediatric-capable facility review, and reporting by age-based program category.

## Fatality Rate

This view shows fatality rate by hospital and month.

Primary use:

Review aggregate outcome trends and identify reporting periods that may require further analyst review.

Important boundary:

Fatality rate is an aggregate reporting measure. It is not a patient-level outcome prediction model.

## ICU And Ventilator

This view summarizes critical care utilization.

Key measures:

- ICU cases
- Ventilator cases
- ICU rate
- Ventilator rate
- Average ICU length of stay
- Average ventilator days

Primary use:

Monitor care intensity and resource utilization by facility and reporting period.

## ED And EMS

This view summarizes emergency department and EMS timing.

Key measures:

- ED minutes
- EMS response minutes
- EMS scene minutes
- EMS transport minutes
- Total pre-hospital minutes
- EMS recorded-value completeness

Primary use:

Review emergency response timing, transport patterns, and EMS documentation quality.

## Claims Utilization

This view summarizes claims activity.

Key measures:

- Claim count
- Allowed amount
- Paid amount
- Average paid amount
- Claim type
- Payer category

Primary use:

Understand aggregate utilization and payment patterns by claim category.

## Data Quality

The data quality tab summarizes validation findings and reporting anomalies.

Primary use:

Review schema issues, missing values, duplicate records, orphan records, facility submission gaps, and unusual month-over-month reporting changes before trusting downstream reports.

## Assistant

The assistant retrieves context from the local Chroma vector database and uses the local Ollama model to answer questions about:

- Dashboard metrics
- Data validation rules
- Reporting workflow
- PDF report content
- PHI and governance boundaries
- Operational interpretation

Evidence is displayed separately from the final answer so users can review the supporting context without losing readability.
