# Deployment

## GitHub

```bash
git init
git add .
git commit -m "Initial AI-powered healthcare analytics RAG reporting assistant"
git branch -M main
git remote add origin https://github.com/<your-github-username>/ai-powered-healthcare-analytics-rag-reporting-assistant.git
git push -u origin main
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
ollama pull llama3.1:8b
healthcare-pipeline --records 5000 --months 24 --seed 42
healthcare-rag-build
healthcare-rag-eval
streamlit run app/streamlit_app.py
```

Use `healthcare-rag-build --ocr` instead of `healthcare-rag-build` when you want OCR for scanned PDFs/images.

For macOS or Linux activation:

```bash
source .venv/bin/activate
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the main file path to `app/streamlit_app.py`.
5. Deploy.

The app can generate synthetic sample data if processed outputs are missing.
The first RAG run downloads the open-source embedding model used for retrieval. The generation LLM runs through Ollama, so the deployment machine must have Ollama running and the selected model pulled locally. No API key is required.

Streamlit Community Cloud does not provide a persistent local Ollama server by default. For a hosted deployment, use one of these options:

- Host the app on a VM where you can install Ollama.
- Keep the repository local-first and include deployment notes.
- Add a separate cloud LLM provider later only if you are comfortable using an API key.

## Docker

```bash
docker build -t healthcare-analytics .
docker run -p 8501:8501 healthcare-analytics
```

Then open `http://localhost:8501`.
