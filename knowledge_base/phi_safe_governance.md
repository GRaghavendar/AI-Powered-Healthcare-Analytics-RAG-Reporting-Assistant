# Data Privacy And Governance

## Purpose

This document defines the privacy and governance boundaries for the trauma analytics application. The application is designed to operate with non-identifiable sample data and aggregate reporting outputs.

## Included Data Domains

The local reporting environment includes:

- Trauma registry-style records
- EMS records
- Hospital transfer records
- Claims records
- Organ donor registration records
- Hospital master reference data
- Population reference data
- Aggregate metrics
- Extracted public report text
- Documentation and metric definitions

## Excluded Data

The application must not include direct identifiers.

Excluded examples:

- Patient names
- Dates of birth
- Street addresses
- Phone numbers
- Email addresses
- Social Security numbers
- Medical record numbers
- MRNs
- Account numbers
- Real patient-level submissions
- Confidential claims records
- Confidential donor records

## Aggregate Reporting Boundary

Dashboard outputs are aggregate summaries. The assistant should explain measures, trends, workflow, validation rules, and documentation. It should not present or infer individual patient information.

## LLM Boundary

The local LLM receives:

- Retrieved knowledge-base context
- Aggregate metric snapshots
- Extracted public report text
- Optional web or weather evidence when enabled

The local LLM should not receive:

- Patient-level identifiers
- Confidential submissions
- Sensitive free-text notes
- Unreviewed operational data containing PHI

## Web Search Boundary

Web search should only be used for non-sensitive public questions. Do not send patient-level data, names, identifiers, facility-confidential details, or internal notes to web search.

## Analyst Review

LLM output should be reviewed before being used in formal reporting. The assistant supports communication and analysis, but it does not replace data validation, governance review, subject matter expertise, or operational approval.

## Approved Assistant Behavior

The assistant may:

- Explain metric definitions.
- Summarize aggregate dashboard trends.
- Describe validation checks.
- Explain PDF report content that has been indexed.
- Identify what evidence was used.
- State when the local knowledge base does not contain enough information.

The assistant may not:

- Claim access to real patient records.
- Make clinical treatment recommendations.
- Provide legal determinations.
- Treat aggregate metrics as patient-level predictions.
- Invent source data that is not present in the knowledge base.
