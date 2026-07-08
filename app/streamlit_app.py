"""Streamlit dashboard for synthetic healthcare trauma analytics."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_analytics.pipeline import run_pipeline  # noqa: E402
from healthcare_analytics.rag import answer_question, build_chroma_index, evidence_snippet  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
UPLOAD_DIR = PROJECT_ROOT / "knowledge_base" / "uploads"

APP_TITLE = "AI-Powered Healthcare Analytics & RAG Reporting Assistant"
SAMPLE_QUESTIONS = [
    "What are the main trauma metrics in this project?",
    "Which trauma center has the highest case volume?",
    "Summarize the latest executive overview.",
    "What data quality issues were found?",
    "Explain the ICU and ventilator utilization trends.",
    "List facilities with reporting anomalies.",
]
PAGES = [
    "Executive Overview",
    "Trauma Center Comparison",
    "Adult vs Pediatric",
    "Fatality Rate",
    "ICU & Ventilator",
    "ED/EMS",
    "Claims Utilization",
    "Data Quality",
    "LLM Executive Summary",
    "RAG Assistant",
]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_outputs() -> None:
    if (PROCESSED_DIR / "executive_overview.csv").exists():
        return
    st.warning("Synthetic analytics outputs have not been generated yet.")
    if st.button("Generate Synthetic Data"):
        run_pipeline(PROJECT_ROOT, records=5000, months=24, seed=42)
        st.cache_data.clear()
        st.rerun()
    st.stop()


def metric_row(latest: pd.Series) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trauma Cases", f"{int(latest['total_cases']):,}")
    col2.metric("Fatality Rate", f"{latest['fatality_rate']:.2f}%")
    col3.metric("ICU Rate", f"{latest['icu_rate']:.2f}%")
    col4.metric("Ventilator Rate", f"{latest['ventilator_rate']:.2f}%")
    col5.metric("Claims Paid", f"${latest['total_paid_amount']:,.0f}")


def insight(text: str) -> None:
    st.markdown(f"**Key Insight:** {text}")


def read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_sidebar() -> str:
    st.sidebar.title("Project Controls")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Synthetic data demo. No PHI, patient records, or real hospital submissions are used.")
    if st.sidebar.button("Regenerate Synthetic Data"):
        run_pipeline(PROJECT_ROOT, records=5000, months=24, seed=42)
        st.cache_data.clear()
        st.rerun()
    return page


def render_executive_overview(overview: pd.DataFrame, forecast: pd.DataFrame) -> None:
    chart = overview.set_index("month")[["total_cases", "adult_cases", "pediatric_cases"]]
    st.line_chart(chart)
    latest = overview.sort_values("month").iloc[-1]
    insight(
        f"The latest synthetic reporting month has {int(latest['total_cases'])} trauma cases. "
        f"Adult cases make up most reported volume, while pediatric volume remains lower and steadier."
    )
    if not forecast.empty:
        st.subheader("Case Volume Forecast")
        st.dataframe(forecast, use_container_width=True, hide_index=True)


def render_trauma_center_comparison(hospital: pd.DataFrame) -> None:
    if hospital.empty:
        st.info("Hospital comparison output is not available yet.")
        return
    hospital_sorted = hospital.sort_values("total_cases", ascending=False)
    chart = (
        alt.Chart(hospital_sorted)
        .mark_bar()
        .encode(
            x=alt.X("total_cases:Q", title="Total Cases"),
            y=alt.Y("hospital_name:N", sort="-x", title="Hospital Name"),
            tooltip=[
                alt.Tooltip("hospital_name:N", title="Hospital"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("trauma_level:N", title="Trauma Level"),
                alt.Tooltip("total_cases:Q", title="Total Cases"),
                alt.Tooltip("fatality_rate:Q", title="Fatality Rate", format=".2f"),
                alt.Tooltip("icu_rate:Q", title="ICU Rate", format=".2f"),
            ],
        )
        .properties(height=max(320, len(hospital_sorted) * 38))
    )
    st.altair_chart(chart, use_container_width=True)
    top = hospital_sorted.iloc[0]
    insight(
        f"{top['hospital_name']} has the highest synthetic case volume with "
        f"{int(top['total_cases'])} cases. Horizontal bars keep full hospital names readable."
    )
    st.dataframe(hospital_sorted, use_container_width=True, hide_index=True)


def render_adult_pediatric(demographic: pd.DataFrame) -> None:
    adult_pediatric = demographic.loc[demographic["segment_type"] == "adult_pediatric"]
    st.bar_chart(adult_pediatric.set_index("segment_value")["total_cases"])
    if not adult_pediatric.empty:
        top = adult_pediatric.sort_values("total_cases", ascending=False).iloc[0]
        insight(f"{top['segment_value']} cases represent the larger share of synthetic trauma volume.")
    st.dataframe(adult_pediatric, use_container_width=True, hide_index=True)


def render_fatality_rate(facility_monthly: pd.DataFrame, hospital: pd.DataFrame) -> None:
    if facility_monthly.empty:
        st.info("Facility monthly output is not available yet.")
        return
    hospital_options = sorted(facility_monthly["hospital_name"].dropna().unique())
    if not hospital_options:
        st.info("No hospital names are available for fatality trend analysis.")
        return

    default_hospital = hospital.sort_values("total_cases", ascending=False).iloc[0]["hospital_name"] if not hospital.empty else hospital_options[0]
    default_index = hospital_options.index(default_hospital) if default_hospital in hospital_options else 0
    selected_hospital = st.selectbox("Select Hospital", hospital_options, index=default_index)

    average = facility_monthly.groupby("month", as_index=False)["fatality_rate"].mean()
    average = average.rename(columns={"fatality_rate": "All hospitals average"})
    selected = facility_monthly.loc[
        facility_monthly["hospital_name"] == selected_hospital,
        ["month", "fatality_rate"],
    ].rename(columns={"fatality_rate": selected_hospital})
    fatality_chart = average.merge(selected, on="month", how="left")
    st.line_chart(fatality_chart.set_index("month"))

    selected_avg = selected[selected_hospital].mean()
    overall_avg = average["All hospitals average"].mean()
    insight(
        f"{selected_hospital} averages {selected_avg:.2f}% across the synthetic period, "
        f"compared with {overall_avg:.2f}% across all hospitals."
    )
    st.dataframe(
        facility_monthly.loc[
            facility_monthly["hospital_name"] == selected_hospital,
            ["month", "hospital_name", "total_cases", "fatality_rate"],
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_icu_ventilator(icu_vent: pd.DataFrame) -> None:
    monthly = icu_vent.groupby("month")[["icu_cases", "ventilator_cases"]].sum()
    st.line_chart(monthly)
    if not monthly.empty:
        latest = monthly.iloc[-1]
        insight(
            f"The latest month includes {int(latest['icu_cases'])} ICU cases and "
            f"{int(latest['ventilator_cases'])} ventilator cases in the synthetic output."
        )
    st.dataframe(icu_vent, use_container_width=True, hide_index=True)


def render_ed_ems(ed_ems: pd.DataFrame) -> None:
    region = ed_ems.groupby("region")[["avg_ed_minutes", "avg_response_minutes", "avg_total_pre_hospital_minutes"]].mean()
    st.bar_chart(region)
    if not region.empty:
        fastest = region["avg_response_minutes"].idxmin()
        insight(f"{fastest} has the lowest average EMS response time in the current synthetic summary.")
    st.dataframe(ed_ems, use_container_width=True, hide_index=True)


def render_claims(claims: pd.DataFrame) -> None:
    claims_by_type = claims.groupby("claim_type")[["claim_count", "paid_amount"]].sum()
    st.bar_chart(claims_by_type["paid_amount"])
    if not claims_by_type.empty:
        top_claim = claims_by_type.sort_values("paid_amount", ascending=False).iloc[0]
        insight(f"The largest paid amount is associated with {top_claim.name} claims.")
    st.dataframe(claims, use_container_width=True, hide_index=True)


def render_data_quality(dq: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.dataframe(dq, use_container_width=True, hide_index=True)
    if not dq.empty:
        total_rows = int(dq["rows_affected"].sum()) if "rows_affected" in dq.columns else 0
        insight(f"The validation layer found {total_rows:,} affected rows across the current synthetic quality checks.")
    if not anomalies.empty:
        flagged = anomalies.loc[anomalies["anomaly_flag"].astype(str).str.lower().eq("true")]
        st.subheader("Reporting Anomalies")
        st.dataframe(flagged, use_container_width=True, hide_index=True)


def render_llm_summary() -> None:
    summary_text = read_markdown(REPORTS_DIR / "executive_summary.md")
    narrative_text = read_markdown(REPORTS_DIR / "facility_narratives.md")
    if summary_text:
        st.markdown(summary_text)
    if narrative_text:
        with st.expander("Facility Narratives"):
            st.markdown(narrative_text)


def retrieval_confidence(result) -> str:
    if not result.sources:
        return "Not applicable"
    best_score = min(source.score for source in result.sources)
    if best_score <= 0.35:
        return "High"
    if best_score <= 0.9:
        return "Medium"
    return "Review sources"


def render_source_table(result) -> None:
    if not result.sources:
        st.caption("No local RAG sources were used for this answer.")
        return

    rows = []
    for idx, source in enumerate(result.sources, start=1):
        rows.append(
            {
                "source_number": idx,
                "source_type": source.source_type,
                "section": source.section,
                "source": source.source,
                "retrieval_score": round(source.score, 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for idx, source in enumerate(result.sources, start=1):
        label = f"Source {idx}: {source.section} | {source.source}"
        with st.expander(label):
            st.write(evidence_snippet(source, 900))


def render_rag_assistant() -> None:
    st.subheader("RAG Assistant")

    if "rag_question" not in st.session_state:
        st.session_state["rag_question"] = "What are the main trauma metrics and how is PHI protected?"
    if "rag_result" not in st.session_state:
        st.session_state["rag_result"] = None

    st.markdown("#### Sample Questions")
    sample_cols = st.columns(3)
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        if sample_cols[idx % 3].button(sample, key=f"sample_question_{idx}", use_container_width=True):
            st.session_state["rag_question"] = sample

    with st.expander("Advanced RAG Settings"):
        setting_col1, setting_col2 = st.columns(2)
        with setting_col1:
            embedding_model = st.text_input(
                "Embedding model",
                value="sentence-transformers/all-MiniLM-L6-v2",
            )
            llm_model = st.text_input("Ollama model", value="llama3.1:8b")
            top_k = st.slider("Retrieved sources", min_value=2, max_value=12, value=6)
        with setting_col2:
            use_web = st.checkbox("Use web search for external questions", value=True)
            use_weather = st.checkbox("Use live weather lookup", value=True)
            use_ocr = st.checkbox("Use OCR while indexing PDFs/images", value=False)
        uploaded_files = st.file_uploader(
            "Add PDFs or images to knowledge base",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )
        upload_col, index_col = st.columns(2)
        with upload_col:
            if uploaded_files and st.button("Save Uploaded Files", use_container_width=True):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                for uploaded in uploaded_files:
                    target = UPLOAD_DIR / uploaded.name
                    target.write_bytes(uploaded.getbuffer())
                st.success(f"Saved {len(uploaded_files)} file(s). Rebuild the Chroma index next.")
        with index_col:
            if st.button("Build Chroma Index", use_container_width=True):
                try:
                    result = build_chroma_index(
                        project_root=PROJECT_ROOT,
                        embedding_model_name=embedding_model,
                        reset=True,
                        use_ocr=use_ocr,
                    )
                    st.success(f"Indexed {result['chunk_count']} chunks.")
                    st.caption(
                        f"PDFs: {result['pdf_files']} | Pages: {result['pages_extracted']} | "
                        f"OCR pages/images: {result['ocr_pages']}"
                    )
                    for warning in result.get("ingestion_warnings", []):
                        st.warning(warning)
                except Exception as exc:
                    st.error(str(exc))

    question = st.text_area("Question", key="rag_question", height=120)
    ask_clicked = st.button("Ask Assistant", type="primary")
    if ask_clicked:
        try:
            st.session_state["rag_result"] = answer_question(
                question=question,
                project_root=PROJECT_ROOT,
                embedding_model_name=embedding_model,
                llm_model_name=llm_model,
                top_k=top_k,
                use_web=use_web,
                use_weather=use_weather,
            )
        except Exception as exc:
            st.session_state["rag_result"] = None
            st.error(str(exc))

    result = st.session_state.get("rag_result")
    if not result:
        return

    st.markdown("### Assistant Answer")
    st.write(result.answer)

    mode_col, source_col, web_col, confidence_col = st.columns(4)
    mode_col.metric("Mode Used", result.mode)
    source_col.metric("Local Sources", len(result.sources))
    web_col.metric("Web Results", len(result.web_results or []))
    confidence_col.metric("Retrieval Confidence", retrieval_confidence(result))

    st.markdown("### Sources Retrieved")
    render_source_table(result)

    if result.web_results:
        st.markdown("### Web Evidence")
        for idx, item in enumerate(result.web_results, start=1):
            with st.expander(f"Web result {idx}: {item.title}"):
                st.write(item.snippet)
                st.write(item.url)

    if result.weather_result:
        st.markdown("### Weather Evidence")
        with st.expander(f"Open-Meteo: {result.weather_result.location}"):
            st.write(f"Condition: {result.weather_result.condition}")
            st.write(f"Temperature: {result.weather_result.temperature_f} F")
            st.write(f"Feels like: {result.weather_result.apparent_temperature_f} F")
            st.write(f"Humidity: {result.weather_result.humidity_percent}%")
            st.write(f"Wind speed: {result.weather_result.wind_speed_mph} mph")
            st.write(result.weather_result.source_url)

    st.markdown("### Trace / Mode Used")
    with st.expander("LangGraph Trace"):
        for step in result.trace:
            st.write(step)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_outputs()

    overview = load_csv("executive_overview")
    hospital = load_csv("hospital_comparison")
    demographic = load_csv("demographic_analytics")
    facility_monthly = load_csv("facility_monthly_submissions")
    icu_vent = load_csv("icu_ventilator_metrics")
    ed_ems = load_csv("ed_ems_summary")
    claims = load_csv("claims_utilization")
    dq = load_csv("data_quality_summary")
    anomalies = load_csv("reporting_anomalies")
    forecast = load_csv("case_volume_forecast")

    st.title(APP_TITLE)
    st.markdown(
        "An end-to-end synthetic healthcare analytics platform for trauma reporting, "
        "data quality monitoring, forecasting, anomaly detection, and RAG-based executive question answering."
    )
    st.info("Synthetic Data Demo | No PHI | No Real Patient Records | No Real Hospital Submissions")

    latest = overview.sort_values("month").iloc[-1]
    metric_row(latest)
    st.divider()

    page = render_sidebar()
    if page == "Executive Overview":
        render_executive_overview(overview, forecast)
    elif page == "Trauma Center Comparison":
        render_trauma_center_comparison(hospital)
    elif page == "Adult vs Pediatric":
        render_adult_pediatric(demographic)
    elif page == "Fatality Rate":
        render_fatality_rate(facility_monthly, hospital)
    elif page == "ICU & Ventilator":
        render_icu_ventilator(icu_vent)
    elif page == "ED/EMS":
        render_ed_ems(ed_ems)
    elif page == "Claims Utilization":
        render_claims(claims)
    elif page == "Data Quality":
        render_data_quality(dq, anomalies)
    elif page == "LLM Executive Summary":
        render_llm_summary()
    elif page == "RAG Assistant":
        render_rag_assistant()


if __name__ == "__main__":
    main()
