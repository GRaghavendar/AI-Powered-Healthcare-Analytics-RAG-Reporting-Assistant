"""LangGraph agent workflow for routing assistant questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from .config import DEFAULT_CHROMA_COLLECTION, DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, PROJECT_ROOT
from .local_llm import LocalLLM, LocalLLMConfig
from .weather_tool import extract_weather_location, fetch_current_weather


class AssistantState(TypedDict, total=False):
    question: str
    project_root: str
    collection_name: str
    embedding_model_name: str
    llm_model_name: str
    top_k: int
    use_web: bool
    use_weather: bool
    llm: Any
    trace: list[str]
    answer: str
    prompt: str
    mode: str
    sources: list[Any]
    contexts: list[Any]
    web_results: list[Any]
    weather_result: Any
    use_rag: bool
    system_prompt: str
    query_scope: str


def classify_question(state: AssistantState) -> AssistantState:
    from .rag import build_general_prompt, classify_query_scope

    question = state["question"]
    query_scope = classify_query_scope(question)
    trace = [*state.get("trace", []), f"classify_question: scope={query_scope}"]
    if query_scope == "greeting":
        return {
            **state,
            "query_scope": query_scope,
            "prompt": build_general_prompt(question),
            "answer": (
                "Hi! I can help you understand the trauma analytics dashboard, explain the RAG workflow, "
                "summarize metrics, review data quality checks, or answer general questions."
            ),
            "mode": "greeting",
            "sources": [],
            "web_results": [],
            "weather_result": None,
            "trace": [*trace, "classify_question: greeting"],
        }
    return {**state, "query_scope": query_scope, "trace": trace}


def route_after_classification(state: AssistantState) -> Literal["finish", "retrieve_rag", "weather_tool", "web_search", "compose_prompt"]:
    query_scope = state.get("query_scope")
    if state.get("mode") == "greeting" or query_scope == "greeting":
        return "finish"
    if query_scope == "internal":
        return "retrieve_rag"
    if query_scope == "weather":
        return "weather_tool"
    if query_scope == "external":
        return "web_search"
    return "compose_prompt"


def retrieve_rag(state: AssistantState) -> AssistantState:
    from .rag import retrieve_context

    try:
        contexts = retrieve_context(
            question=state["question"],
            project_root=state["project_root"],
            collection_name=state["collection_name"],
            embedding_model_name=state["embedding_model_name"],
            top_k=state["top_k"],
        )
        source_types = sorted({getattr(context, "source_type", "unknown") for context in contexts})
        trace = [*state.get("trace", []), f"retrieve_rag: retrieved {len(contexts)} chunks; source_types={source_types}"]
    except Exception:
        contexts = []
        trace = [*state.get("trace", []), "retrieve_rag: Chroma index unavailable or empty"]
    return {**state, "contexts": contexts, "trace": trace}


def grade_context(state: AssistantState) -> AssistantState:
    from .rag import has_useful_rag_context

    use_rag = has_useful_rag_context(state["question"], state.get("contexts", []))
    trace = [*state.get("trace", []), f"grade_context: use_rag={use_rag}"]
    return {**state, "use_rag": use_rag, "trace": trace}


def weather_tool(state: AssistantState) -> AssistantState:
    location = extract_weather_location(state["question"])
    trace = [*state.get("trace", [])]
    if not location:
        return {**state, "weather_result": None, "trace": [*trace, "weather_tool: weather question detected but no location was found"]}
    try:
        weather_result = fetch_current_weather(location)
        trace.append(f"weather_tool: retrieved current weather for {weather_result.location}")
    except Exception as exc:
        weather_result = None
        trace.append(f"weather_tool: unavailable or failed ({exc})")
    return {**state, "weather_result": weather_result, "trace": trace}


def web_search(state: AssistantState) -> AssistantState:
    from .web_search import search_duckduckgo

    weather_result = state.get("weather_result")
    web_results = []
    trace = [*state.get("trace", [])]
    query_scope = state.get("query_scope")
    should_search = state.get("use_web", True) and (query_scope == "external" or (query_scope == "weather" and not weather_result))
    if should_search:
        try:
            web_results = search_duckduckgo(state["question"], max_results=4)
            trace.append(f"web_search: retrieved {len(web_results)} web snippets")
        except Exception:
            trace.append("web_search: unavailable or failed")
    return {**state, "web_results": web_results, "trace": trace}


def compose_prompt(state: AssistantState) -> AssistantState:
    from .rag import build_general_prompt, build_hybrid_prompt, load_system_prompt

    use_rag = bool(state.get("use_rag"))
    web_results = state.get("web_results", [])
    weather_result = state.get("weather_result")
    if use_rag or web_results or weather_result:
        mode_parts: list[str] = []
        if use_rag:
            mode_parts.append("rag")
        if weather_result:
            mode_parts.append("weather")
        if web_results:
            mode_parts.append("web")
        mode = "+".join(mode_parts)
        prompt = build_hybrid_prompt(
            state["question"],
            state.get("contexts", []) if use_rag else [],
            web_results,
            weather_result=weather_result,
            mode=mode,
        )
    else:
        mode = "external_llm" if state.get("query_scope") == "external" else "general_llm"
        prompt = build_general_prompt(state["question"])

    trace = [*state.get("trace", []), f"compose_prompt: mode={mode}"]
    return {
        **state,
        "mode": mode,
        "prompt": prompt,
        "system_prompt": load_system_prompt(state["project_root"]),
        "trace": trace,
    }


def generate_answer(state: AssistantState) -> AssistantState:
    generator = state.get("llm") or LocalLLM(LocalLLMConfig(model_name=state["llm_model_name"]))
    answer = generator.generate(state["prompt"], system_prompt=state.get("system_prompt"))
    trace = [*state.get("trace", []), "generate_answer: completed with local Ollama/HF model"]
    return {**state, "answer": answer, "trace": trace}


def finish(state: AssistantState) -> AssistantState:
    return state


def build_agent_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Run: pip install -r requirements.txt") from exc

    graph = StateGraph(AssistantState)
    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("grade_context", grade_context)
    graph.add_node("weather_tool", weather_tool)
    graph.add_node("web_search", web_search)
    graph.add_node("compose_prompt", compose_prompt)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("finish", finish)

    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {
            "finish": "finish",
            "retrieve_rag": "retrieve_rag",
            "weather_tool": "weather_tool",
            "web_search": "web_search",
            "compose_prompt": "compose_prompt",
        },
    )
    graph.add_edge("retrieve_rag", "grade_context")
    graph.add_edge("grade_context", "compose_prompt")
    graph.add_edge("weather_tool", "web_search")
    graph.add_edge("web_search", "compose_prompt")
    graph.add_edge("compose_prompt", "generate_answer")
    graph.add_edge("generate_answer", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_agent_graph(
    question: str,
    project_root: str | Path = PROJECT_ROOT,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
    llm: LocalLLM | None = None,
    use_web: bool = True,
    use_weather: bool = True,
):
    from .rag import RagAnswer

    graph = build_agent_graph()
    state = graph.invoke(
        {
            "question": question,
            "project_root": str(Path(project_root)),
            "collection_name": collection_name,
            "embedding_model_name": embedding_model_name,
            "llm_model_name": llm_model_name,
            "top_k": top_k,
            "llm": llm,
            "use_web": use_web,
            "use_weather": use_weather,
            "trace": [],
            "contexts": [],
            "web_results": [],
            "sources": [],
        }
    )
    use_rag = bool(state.get("use_rag"))
    contexts = state.get("contexts", [])
    return RagAnswer(
        question=question,
        answer=state.get("answer", ""),
        sources=contexts if use_rag else [],
        prompt=state.get("prompt", ""),
        model_name=llm_model_name,
        embedding_model=embedding_model_name,
        mode=state.get("mode", "general_llm"),
        web_results=state.get("web_results", []),
        weather_result=state.get("weather_result"),
        trace=state.get("trace", []),
    )
