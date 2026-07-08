FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY docs ./docs
COPY knowledge_base ./knowledge_base
COPY data ./data

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
