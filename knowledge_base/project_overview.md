# Project Overview

## System Name

AI-Powered Healthcare Analytics & RAG Reporting Assistant

## Purpose

The application supports trauma reporting analysis, data quality review, aggregate metric generation, document retrieval, and natural-language question answering. It combines a structured analytics pipeline with a local retrieval-augmented assistant.

## Core Capabilities

- Build trauma registry-style reporting datasets.
- Validate source files before metric generation.
- Create dashboard-ready aggregate outputs.
- Detect unusual facility reporting changes.
- Forecast near-term case volume.
- Extract text from PDF reports and optional OCR-supported files.
- Index documentation, report extracts, and metric snapshots in Chroma.
- Answer user questions with a local Ollama model and retrieved evidence.
- Use optional web, weather, and document tools for questions that require live or external context.

## Data Domains

The application works with the following reporting domains:

- Trauma registry records
- EMS records
- Hospital transfers
- Claims utilization
- Hospital master reference data
- Population reference data
- Public report extracts

## Analytics Workflow

1. Load or generate local reporting data.
2. Validate schema, keys, missing values, allowed values, numeric ranges, and cross-dataset relationships.
3. Build aggregate reporting metrics.
4. Run anomaly detection for hospital-month submission changes.
5. Forecast near-term case volume.
6. Write processed CSV outputs and reporting summaries.
7. Convert metric snapshots and document extracts into markdown.
8. Build the Chroma vector index.
9. Route user questions through the assistant workflow.
10. Return a clear answer with evidence available for review.

## Assistant Workflow

The assistant follows a tool-routing pattern:

- Use Chroma RAG for internal metrics, dashboard questions, PDF report questions, validation rules, and governance topics.
- Use the local Ollama model for general explanation and final response generation.
- Use web search only for public external information when enabled.
- Use Open-Meteo for current weather questions.
- State clearly when the local knowledge base does not contain enough information.

## Governance Boundary

The application is designed around non-identifiable sample data and aggregate outputs. It does not require API keys for the local LLM or vector database. The assistant should not claim access to real patient-level data, confidential submissions, or protected health information.

## Expected Users

Expected users include analysts, data science practitioners, reporting teams, and technical reviewers who need to understand the workflow, metrics, data quality checks, and assistant behavior.
