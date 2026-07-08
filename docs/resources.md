# Project Resources

| Resource Name | Path | Description |
|---|---|---|
| Project README | `README.md` | Repository overview, setup, run, and deployment notes. |
| Architecture diagram | `docs/architecture.md` | Mermaid diagram and component map. |
| Workflow diagram | `docs/workflow.md` | End-to-end pipeline flow. |
| Data dictionary | `docs/data_dictionary.md` | Raw and processed file definitions. |
| PHI-safe design | `docs/phi_safe_design.md` | Data privacy and governance principles. |
| Deployment guide | `docs/deployment.md` | GitHub, Streamlit Cloud, and Docker deployment steps. |
| Agent workflow | `docs/agent_workflow.md` | LangGraph conditional router graph for classify, retrieve, tool routing, prompt, and answer. |
| PDF/OCR ingestion guide | `docs/pdf_ocr_ingestion.md` | How PDFs/images are extracted and indexed into Chroma. |
| LLM system prompt | `prompts/assistant_system_prompt.md` | Governs chatbot behavior, RAG rules, web rules, and fallback behavior. |
| RAG knowledge base | `knowledge_base/` | Metric definitions, validation rules, governance, dashboard guide, and project overview. |
| Pipeline CLI | `src/healthcare_analytics/pipeline.py` | End-to-end synthetic data and reporting pipeline. |
| Document ingestion | `src/healthcare_analytics/document_ingestion.py` | Extracts PDF text and optional OCR text into generated markdown. |
| LangGraph agent | `src/healthcare_analytics/agent_graph.py` | Defines graph state, nodes, conditional edges, and compiled assistant workflow. |
| RAG pipeline | `src/healthcare_analytics/rag.py` | Chroma index builder, retriever, agent prompt builder, and CLI Q&A. |
| RAG evaluation | `src/healthcare_analytics/rag_evaluation.py` | Runs sample stakeholder questions and writes sourced answers to JSON. |
| Web search helper | `src/healthcare_analytics/web_search.py` | Optional DuckDuckGo search snippets for hybrid assistant mode. |
| Weather helper | `src/healthcare_analytics/weather_tool.py` | No-key Open-Meteo lookup for current weather questions. |
| Local LLM wrapper | `src/healthcare_analytics/local_llm.py` | Ollama wrapper for local open-source model inference. |
| Streamlit app | `app/streamlit_app.py` | Interactive dashboard. |
| Unit tests | `tests/test_pipeline.py` | Smoke tests for generation, validation, metrics, modeling, and reports. |
