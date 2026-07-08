"""Project constants and synthetic reference data."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("HEALTHCARE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_store" / "chroma"

RANDOM_SEED = 42
DEFAULT_MONTHS = 24
DEFAULT_RECORDS = 5000
DEFAULT_END_MONTH = "2025-12-01"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "HEALTHCARE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_LLM_PROVIDER = os.getenv("HEALTHCARE_LLM_PROVIDER", "ollama")
DEFAULT_OLLAMA_BASE_URL = os.getenv("HEALTHCARE_OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv(
    "HEALTHCARE_LLM_MODEL",
    os.getenv("HEALTHCARE_OLLAMA_MODEL", "llama3.1:8b"),
)
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("HEALTHCARE_LLM_MAX_TOKENS", "-1"))
DEFAULT_LLM_CONTEXT_TOKENS = int(os.getenv("HEALTHCARE_LLM_CONTEXT_TOKENS", "8192"))
DEFAULT_CHROMA_COLLECTION = os.getenv("HEALTHCARE_CHROMA_COLLECTION", "healthcare_rag")

HOSPITAL_RECORDS = [
    {
        "hospital_id": "H001",
        "hospital_name": "North River Trauma Center",
        "region": "Capital",
        "trauma_level": "Level I",
        "bed_count": 620,
        "pediatric_capable": True,
    },
    {
        "hospital_id": "H002",
        "hospital_name": "Lakeview Regional Medical",
        "region": "Western",
        "trauma_level": "Level II",
        "bed_count": 410,
        "pediatric_capable": True,
    },
    {
        "hospital_id": "H003",
        "hospital_name": "Hudson Valley Trauma Institute",
        "region": "Hudson",
        "trauma_level": "Level I",
        "bed_count": 540,
        "pediatric_capable": True,
    },
    {
        "hospital_id": "H004",
        "hospital_name": "Metro East Medical Center",
        "region": "Metro",
        "trauma_level": "Level II",
        "bed_count": 365,
        "pediatric_capable": False,
    },
    {
        "hospital_id": "H005",
        "hospital_name": "Central County Hospital",
        "region": "Central",
        "trauma_level": "Level III",
        "bed_count": 250,
        "pediatric_capable": False,
    },
    {
        "hospital_id": "H006",
        "hospital_name": "Harborview Emergency Hospital",
        "region": "Metro",
        "trauma_level": "Level I",
        "bed_count": 700,
        "pediatric_capable": True,
    },
    {
        "hospital_id": "H007",
        "hospital_name": "Pine Hills Medical Center",
        "region": "Northern",
        "trauma_level": "Level III",
        "bed_count": 215,
        "pediatric_capable": False,
    },
    {
        "hospital_id": "H008",
        "hospital_name": "South Bay Trauma Center",
        "region": "Long Island",
        "trauma_level": "Level II",
        "bed_count": 455,
        "pediatric_capable": True,
    },
]

REGION_POPULATION = {
    "Capital": 1080000,
    "Western": 1410000,
    "Hudson": 1670000,
    "Metro": 4200000,
    "Central": 980000,
    "Northern": 610000,
    "Long Island": 2830000,
}

RACE_CATEGORIES = [
    "American Indian or Alaska Native",
    "Asian",
    "Black or African American",
    "Hispanic or Latino",
    "White",
    "Other or Unknown",
]

GENDER_CATEGORIES = ["Female", "Male", "Unknown"]
AGE_GROUPS = ["0-17", "18-34", "35-49", "50-64", "65+"]
INJURY_TYPES = [
    "Fall",
    "Motor vehicle",
    "Assault",
    "Burn",
    "Sports injury",
    "Other blunt trauma",
]
CLAIM_TYPES = ["Inpatient", "Outpatient", "Ambulatory Surgery", "Emergency Department"]
PAYER_CATEGORIES = ["Commercial", "Medicaid", "Medicare", "Self Pay", "Other"]
REGISTRATION_LOCATIONS = ["DMV", "Online", "Hospital", "Community Event", "Other"]
