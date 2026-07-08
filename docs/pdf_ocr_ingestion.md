# PDF And OCR Ingestion

The knowledge base can include markdown, PDFs, and images.

## What Happens During Index Build

1. Markdown files are read directly.
2. PDF files in `knowledge_base/` are extracted page by page with PyMuPDF.
3. If `--ocr` is enabled, scanned PDF pages and image files are passed to Tesseract through `pytesseract`.
4. Extracted text is saved as markdown under `knowledge_base/generated/extracted_documents/`.
5. The generated markdown is chunked, embedded, and stored in Chroma.

## Commands

Extract documents only:

```powershell
healthcare-ingest-docs --ocr
```

Build the full Chroma index:

```powershell
healthcare-rag-build --ocr
```

## Notes

- Selectable PDF text usually works without OCR.
- Scanned PDFs and images require OCR.
- `pytesseract` is the Python wrapper. Windows also needs the Tesseract desktop program installed separately.
- OCR can read chart titles, labels, legends, and table text. It does not fully understand every chart pattern like a vision-language model would.
