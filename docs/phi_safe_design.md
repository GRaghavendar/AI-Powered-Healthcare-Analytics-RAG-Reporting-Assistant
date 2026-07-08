# PHI-Safe Design

This document defines privacy and governance boundaries for the local trauma analytics application.

## Rules Used

- Use synthetic records only.
- Do not include real healthcare, hospital, patient, claims, donor, EMS, ED, or facility submission data.
- Do not generate direct identifiers such as names, phone numbers, emails, street addresses, dates of birth, Social Security numbers, or medical record numbers.
- Use aggregate metrics for the reporting assistant.
- Treat LLM output as a communication aid, not a clinical decision system.
- Keep human review in the workflow for stakeholder summaries.

## What The Local LLM + RAG Layer Does

- Retrieves relevant project documentation and generated aggregate metric snapshots from Chroma.
- Retrieves extracted PDF/OCR report text when reports are added to the knowledge base.
- Uses a local Ollama open-source LLM to answer stakeholder questions.
- Explains metric changes in plain English.
- Summarizes validation findings.
- Shows source documents separately for auditability.

## What The Local LLM + RAG Layer Does Not Do

- It does not predict patient outcomes.
- It does not recommend clinical treatment.
- It does not ingest patient-level PHI.
- It does not replace analyst review.
- It does not send sensitive data to web search.
