"""PDF, image, and OCR ingestion for the RAG knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SKIP_DIRS = {"generated", "generated_ingestion", "__pycache__"}
FACILITY_CASE_LOAD_REGIONS = {
    "Central NY",
    "Finger Lakes",
    "Hudson Valley",
    "Northeastern NY",
    "Western NY",
    "Bronx",
    "Kings",
    "New York",
    "Queens",
    "Richmond",
    "Nassau",
    "Suffolk",
}
DESIGNATION_VALUES = {"Adult Trauma Center", "Pediatric Trauma Center", "Dual Designation"}


@dataclass
class IngestionReport:
    generated_files: list[Path] = field(default_factory=list)
    pdf_files: int = 0
    image_files: int = 0
    pages_extracted: int = 0
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)


def _safe_stem(path: Path, knowledge_base_dir: Path) -> str:
    rel = path.relative_to(knowledge_base_dir).as_posix()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", rel).strip("_")


def _clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def _normalise_report_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\uf0d8": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _report_lines(text: str) -> list[str]:
    lines = []
    for line in _normalise_report_text(text).splitlines():
        cleaned = " ".join(line.split()).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _is_count_token(value: str) -> bool:
    return value == "." or bool(re.fullmatch(r"\d[\d,]*", value))


def _parse_count(value: str) -> int | None:
    if value == ".":
        return None
    return int(value.replace(",", ""))


def _format_count(value: object) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _table_cell(value: object) -> str:
    return str(value).replace("|", "/")


def _iter_external_documents(knowledge_base_dir: Path) -> list[Path]:
    if not knowledge_base_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(knowledge_base_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(knowledge_base_dir).parts):
            continue
        if path.suffix.lower() in PDF_EXTENSIONS | IMAGE_EXTENSIONS:
            files.append(path)
    return files


def _parse_audit_facility_summary(text: str) -> list[dict[str, object]]:
    """Parse annual facility submission rows from the audit PDF extract."""

    summary_text = text.split("## Page 3", 1)[0]
    lines = _report_lines(summary_text)
    rows: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        inline_match = re.fullmatch(r"(\d{4,5}\*?)\s+(.+)", line)
        if inline_match:
            pfi = inline_match.group(1)
            name = inline_match.group(2)
            value_start = i + 1
        elif re.fullmatch(r"\d{4,5}\*?", line) and i + 1 < len(lines):
            pfi = line
            name = lines[i + 1]
            value_start = i + 2
        else:
            i += 1
            continue

        if name in {"Hospital Name", "Facility Name"} or name.startswith("PFI "):
            i += 1
            continue

        values = lines[value_start : value_start + 12]
        if len(values) < 12 or not all(_is_count_token(value) for value in values):
            i += 1
            continue

        counts = [_parse_count(value) for value in values]
        rows.append(
            {
                "pfi": pfi,
                "hospital_name": name,
                "adult_2022": counts[0],
                "pediatric_2022": counts[1],
                "adult_2023": counts[2],
                "pediatric_2023": counts[3],
                "adult_2024": counts[4],
                "pediatric_2024": counts[5],
                "adult_2025": counts[6],
                "pediatric_2025": counts[7],
                "total_2022": counts[8],
                "total_2023": counts[9],
                "total_2024": counts[10],
                "total_2025": counts[11],
            }
        )
        i = value_start + 12
    return rows


def _parse_facility_case_loads(text: str) -> list[dict[str, object]]:
    """Parse 2016-2020 facility case-load rows from the trauma report extract."""

    lines = _report_lines(text)
    rows: list[dict[str, object]] = []
    active = False
    current_region = ""
    i = 0
    header_values = {
        "Region",
        "Facility",
        "Total",
        "Adult",
        "Pediatric",
        "Facility Casesa",
        "Cases",
        "Cases Verification",
        "Verification",
        "Designation",
    }
    while i < len(lines):
        line = lines[i]
        if "Trauma Center Average Annual Case Loads" in line:
            active = True
            i += 1
            continue
        if active and line.startswith("Statewide Total"):
            break
        if not active:
            i += 1
            continue
        if line in FACILITY_CASE_LOAD_REGIONS:
            current_region = line
            i += 1
            continue
        if line in header_values or line.startswith("#") or line.startswith("- Source"):
            i += 1
            continue
        if line == "Total":
            i += 1
            while i < len(lines) and _is_count_token(lines[i]):
                i += 1
            continue
        if not current_region or i + 5 >= len(lines) or not _is_count_token(lines[i + 1]):
            i += 1
            continue

        verification = lines[i + 4]
        designation = lines[i + 5]
        if not verification.startswith("Level") or designation not in DESIGNATION_VALUES:
            i += 1
            continue
        rows.append(
            {
                "region": current_region,
                "facility": line,
                "total_cases": _parse_count(lines[i + 1]),
                "adult_cases": _parse_count(lines[i + 2]),
                "pediatric_cases": _parse_count(lines[i + 3]),
                "verification": verification,
                "designation": designation,
            }
        )
        i += 6
    return rows


def _write_structured_report_references(knowledge_base_dir: Path, report: IngestionReport) -> None:
    extracted_dir = knowledge_base_dir / "generated" / "extracted_documents"
    generated_dir = knowledge_base_dir / "generated"
    if not extracted_dir.exists():
        return

    audit_rows: list[dict[str, object]] = []
    case_load_rows: list[dict[str, object]] = []
    for path in sorted(extracted_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = path.name.lower()
        if name.startswith("audit_rpt_trauma_case_submitted"):
            audit_rows.extend(_parse_audit_facility_summary(text))
        elif name.startswith("2016-2020_trauma_report"):
            case_load_rows.extend(_parse_facility_case_loads(text))

    generated_dir.mkdir(parents=True, exist_ok=True)
    if audit_rows:
        audit_path = generated_dir / "audit_report_facility_summary.md"
        lines = [
            "# Audit Report Facility Submission Summary",
            "",
            "- Source document: `audit_rpt_trauma_case_submitted.pdf`, pages 1-2.",
            "- Scope: annual adult, pediatric, and total trauma cases submitted by New York State trauma centers for discharge years 2022-2025.",
            "- This structured reference is extracted from the PDF report and is separate from generated synthetic dashboard metrics.",
            "- A dash means the PDF used a dot or blank value for that adult or pediatric field.",
            "",
            "## Facility Summary Table",
            "",
            "| PFI | Hospital Name | 2022 Adult | 2022 Pediatric | 2022 Total | 2023 Adult | 2023 Pediatric | 2023 Total | 2024 Adult | 2024 Pediatric | 2024 Total | 2025 Adult | 2025 Pediatric | 2025 Total |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in audit_rows:
            lines.append(
                "| {pfi} | {hospital_name} | {adult_2022} | {pediatric_2022} | {total_2022} | "
                "{adult_2023} | {pediatric_2023} | {total_2023} | {adult_2024} | {pediatric_2024} | "
                "{total_2024} | {adult_2025} | {pediatric_2025} | {total_2025} |".format(
                    pfi=_table_cell(row["pfi"]),
                    hospital_name=_table_cell(row["hospital_name"]),
                    adult_2022=_format_count(row["adult_2022"]),
                    pediatric_2022=_format_count(row["pediatric_2022"]),
                    total_2022=_format_count(row["total_2022"]),
                    adult_2023=_format_count(row["adult_2023"]),
                    pediatric_2023=_format_count(row["pediatric_2023"]),
                    total_2023=_format_count(row["total_2023"]),
                    adult_2024=_format_count(row["adult_2024"]),
                    pediatric_2024=_format_count(row["pediatric_2024"]),
                    total_2024=_format_count(row["total_2024"]),
                    adult_2025=_format_count(row["adult_2025"]),
                    pediatric_2025=_format_count(row["pediatric_2025"]),
                    total_2025=_format_count(row["total_2025"]),
                )
            )
        lines.extend(["", "## Facility Notes", ""])
        for row in audit_rows:
            lines.append(
                "- {hospital_name} (PFI {pfi}): 2022 total {total_2022}; 2023 total {total_2023}; "
                "2024 total {total_2024}; 2025 total {total_2025}. Adult/pediatric split: "
                "2022 {adult_2022}/{pediatric_2022}, 2023 {adult_2023}/{pediatric_2023}, "
                "2024 {adult_2024}/{pediatric_2024}, 2025 {adult_2025}/{pediatric_2025}.".format(
                    hospital_name=row["hospital_name"],
                    pfi=row["pfi"],
                    total_2022=_format_count(row["total_2022"]),
                    total_2023=_format_count(row["total_2023"]),
                    total_2024=_format_count(row["total_2024"]),
                    total_2025=_format_count(row["total_2025"]),
                    adult_2022=_format_count(row["adult_2022"]),
                    pediatric_2022=_format_count(row["pediatric_2022"]),
                    adult_2023=_format_count(row["adult_2023"]),
                    pediatric_2023=_format_count(row["pediatric_2023"]),
                    adult_2024=_format_count(row["adult_2024"]),
                    pediatric_2024=_format_count(row["pediatric_2024"]),
                    adult_2025=_format_count(row["adult_2025"]),
                    pediatric_2025=_format_count(row["pediatric_2025"]),
                )
            )
        audit_path.write_text("\n".join(lines), encoding="utf-8")
        report.generated_files.append(audit_path)

    if case_load_rows:
        case_load_path = generated_dir / "trauma_report_facility_reference.md"
        lines = [
            "# Trauma Report Facility Reference",
            "",
            "- Source document: `2016-2020_trauma_report.pdf`, pages 11-13.",
            "- Scope: average annual trauma center case loads by region, facility, verification level, and designation.",
            "- This structured reference is extracted from the PDF report and is separate from generated synthetic dashboard metrics.",
            "",
            "## Facility Case Load Table",
            "",
            "| Region | Facility | Total Cases | Adult Cases | Pediatric Cases | Verification | Designation |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for row in case_load_rows:
            lines.append(
                "| {region} | {facility} | {total_cases} | {adult_cases} | {pediatric_cases} | {verification} | {designation} |".format(
                    region=_table_cell(row["region"]),
                    facility=_table_cell(row["facility"]),
                    total_cases=_format_count(row["total_cases"]),
                    adult_cases=_format_count(row["adult_cases"]),
                    pediatric_cases=_format_count(row["pediatric_cases"]),
                    verification=_table_cell(row["verification"]),
                    designation=_table_cell(row["designation"]),
                )
            )
        lines.extend(["", "## Facility Notes", ""])
        for row in case_load_rows:
            lines.append(
                "- {facility} ({region}): average annual total {total_cases}; adult {adult_cases}; pediatric {pediatric_cases}; "
                "{verification}; {designation}.".format(
                    facility=row["facility"],
                    region=row["region"],
                    total_cases=_format_count(row["total_cases"]),
                    adult_cases=_format_count(row["adult_cases"]),
                    pediatric_cases=_format_count(row["pediatric_cases"]),
                    verification=row["verification"],
                    designation=row["designation"],
                )
            )
        case_load_path.write_text("\n".join(lines), encoding="utf-8")
        report.generated_files.append(case_load_path)


def _ocr_pil_image(image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract") from exc
    return pytesseract.image_to_string(image)


def _ocr_pdf_page(page) -> str:
    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for PDF OCR. Run: pip install pymupdf pillow") from exc

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return _ocr_pil_image(image)


def _extract_pdf(path: Path, output_dir: Path, knowledge_base_dir: Path, use_ocr: bool, report: IngestionReport) -> None:
    try:
        import fitz
    except ImportError as exc:
        report.warnings.append(
            f"Skipped {path.name}: PyMuPDF is not installed. Run: pip install pymupdf"
        )
        return

    output_path = output_dir / f"{_safe_stem(path, knowledge_base_dir)}.extracted.md"
    lines = [
        f"# Extracted Document: {path.name}",
        "",
        f"- Source file: `{path.relative_to(knowledge_base_dir).as_posix()}`",
        "- Extraction method: PDF text extraction with optional OCR fallback.",
        "",
    ]

    try:
        with fitz.open(path) as doc:
            report.pdf_files += 1
            for page_number, page in enumerate(doc, start=1):
                text = _clean_text(page.get_text("text"))
                method = "selectable PDF text"
                if use_ocr and len(text) < 80:
                    try:
                        text = _clean_text(_ocr_pdf_page(page))
                        method = "OCR fallback"
                        report.ocr_pages += 1
                    except Exception as exc:
                        report.warnings.append(f"OCR failed for {path.name} page {page_number}: {exc}")

                if not text:
                    text = "[No readable text was extracted from this page.]"
                report.pages_extracted += 1
                lines.extend(
                    [
                        f"## Page {page_number}",
                        "",
                        f"- Source reference: `{path.name}`, page {page_number}",
                        f"- Extraction method: {method}",
                        "",
                        text,
                        "",
                    ]
                )
    except Exception as exc:
        report.warnings.append(f"Skipped {path.name}: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    report.generated_files.append(output_path)


def _extract_image(path: Path, output_dir: Path, knowledge_base_dir: Path, report: IngestionReport) -> None:
    try:
        from PIL import Image
    except ImportError:
        report.warnings.append(f"Skipped {path.name}: Pillow is not installed. Run: pip install pillow")
        return

    output_path = output_dir / f"{_safe_stem(path, knowledge_base_dir)}.ocr.md"
    try:
        with Image.open(path) as image:
            text = _clean_text(_ocr_pil_image(image))
    except Exception as exc:
        report.warnings.append(f"OCR failed for {path.name}: {exc}")
        return

    report.image_files += 1
    report.ocr_pages += 1
    if not text:
        text = "[No readable text was extracted from this image.]"
    lines = [
        f"# OCR Image: {path.name}",
        "",
        f"- Source file: `{path.relative_to(knowledge_base_dir).as_posix()}`",
        "- Extraction method: image OCR",
        "",
        "## Extracted Text",
        "",
        text,
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    report.generated_files.append(output_path)


def extract_knowledge_documents(project_root: str | Path, use_ocr: bool = False) -> IngestionReport:
    """Extract readable PDF/image content into generated markdown files.

    The generated markdown is what Chroma indexes. PDF text extraction is always
    attempted. OCR is only used for scanned PDFs/images when `use_ocr=True`.
    """

    root = Path(project_root).resolve()
    knowledge_base_dir = root / "knowledge_base"
    output_dir = knowledge_base_dir / "generated" / "extracted_documents"
    report = IngestionReport()

    for path in _iter_external_documents(knowledge_base_dir):
        suffix = path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            _extract_pdf(path, output_dir, knowledge_base_dir, use_ocr=use_ocr, report=report)
        elif suffix in IMAGE_EXTENSIONS:
            if use_ocr:
                _extract_image(path, output_dir, knowledge_base_dir, report=report)
            else:
                report.warnings.append(f"Skipped {path.name}: image OCR is disabled. Rebuild with --ocr.")
    _write_structured_report_references(knowledge_base_dir, report)
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract PDFs/images from knowledge_base into markdown for RAG.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--ocr", action="store_true", help="Use OCR for scanned PDFs and image files.")
    args = parser.parse_args(argv)

    report = extract_knowledge_documents(args.project_root, use_ocr=args.ocr)
    print(f"PDF files processed: {report.pdf_files}")
    print(f"Image files processed: {report.image_files}")
    print(f"Pages extracted: {report.pages_extracted}")
    print(f"OCR pages/images: {report.ocr_pages}")
    print(f"Generated files: {len(report.generated_files)}")
    for warning in report.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
