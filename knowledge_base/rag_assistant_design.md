# RAG Assistant Design

## Purpose

The assistant provides natural-language access to the trauma analytics knowledge base, dashboard metric definitions, validation rules, PDF report extracts, and generated aggregate snapshots. It is designed to answer clearly while keeping supporting evidence available for review.

## Architecture Summary

The assistant has six main layers:

1. Knowledge ingestion
2. Text chunking
3. Embedding generation
4. Chroma vector retrieval
5. LangGraph conditional tool routing
6. Local LLM response generation through Ollama

## Knowledge Sources

Indexed knowledge sources include:

- Manually maintained markdown documentation
- Generated aggregate metric snapshots
- Extracted PDF report text
- OCR text from scanned PDFs or images when enabled
- Reporting summaries

## Embedding Model

Default embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

The embedding model converts knowledge chunks into vectors. Those vectors allow semantic search across metric definitions, governance notes, dashboard instructions, generated summaries, and report extracts.

## Vector Database

Vector database:

`Chroma`

Chroma stores the local vector index under `vector_store/chroma`. The vector database is persistent and can be rebuilt whenever knowledge documents or PDF extracts change.

## Local LLM

Default local LLM:

`llama3.1:8b` through Ollama

Ollama runs the generation model locally. Users can switch models by setting `HEALTHCARE_LLM_MODEL`.

Common model options:

- `llama3.1:8b`
- `llama3.2:3b`
- `mistral:7b`
- `qwen2.5:7b`

## Index Build Workflow

1. Write the latest aggregate metric snapshot.
2. Extract text from PDF files in the knowledge base.
3. Run OCR for scanned PDFs and images when OCR is enabled.
4. Load markdown documents and generated extracts.
5. Split text into overlapping chunks.
6. Generate embeddings for each chunk.
7. Store chunks, metadata, and embeddings in Chroma.

## Question Answering Workflow

1. Receive the user question in the LangGraph state.
2. Detect greetings, weather questions, application questions, and current/external questions.
3. Retrieve relevant Chroma context.
4. Grade whether retrieved context is useful.
5. Call the weather tool or web search when needed and enabled.
6. Compose a prompt using the system prompt, retrieved evidence, tool evidence, and user question.
7. Generate the final answer through Ollama.
8. Return the answer, mode, trace, and optional evidence.

## LangGraph Nodes

The compiled graph contains these nodes:

- `classify_question`
- `retrieve_rag`
- `grade_context`
- `weather_tool`
- `web_search`
- `compose_prompt`
- `generate_answer`
- `finish`

Conditional edges route greetings directly to `finish`, weather questions to `weather_tool`, and current or external questions to `web_search` when web search is enabled.

## Assistant Modes

- `greeting`: Responds to simple greetings.
- `rag`: Uses Chroma-retrieved knowledge-base or report evidence.
- `general_llm`: Uses the local LLM when retrieved evidence is not needed.
- `web`: Uses public web snippets when enabled.
- `weather`: Uses Open-Meteo current weather data.
- `rag+web`: Uses local evidence and web evidence together.
- `rag+weather`: Uses local evidence and weather evidence together.

## Answer Rules

The assistant should:

- Answer in natural language.
- Use structured sections for detailed questions.
- Use local RAG evidence for internal application questions.
- Mention when evidence is incomplete.
- Keep source snippets separate from the main answer when displayed in the app.

The assistant should not:

- Only return source paths.
- Treat aggregate metrics as patient-level predictions.
- Invent unavailable data.
- Send sensitive data to web search.
