from __future__ import annotations
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from config.models import LLMAdapter, LLMConfig
from config import settings as C

adapter = LLMAdapter(LLMConfig(C.PROVIDER, C.GENERATION_MODEL, C.EMBED_MODEL))

class AgentState(dict):
    pass

def _hydrate_state(state: AgentState) -> str:
    """Ensure required fields are present regardless of langgraph input format."""

    def _coerce_query(value):  # type: ignore[return-type]
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, HumanMessage):
            return _coerce_query(value.content)
        if isinstance(value, tuple):
            if len(value) == 2 and isinstance(value[0], str):
                role, content = value
                if role.lower() in {"user", "human"}:
                    return _coerce_query(content)
                if not isinstance(content, str):
                    return _coerce_query(content)
                return None
            items = list(value)
            for item in reversed(items):
                coerced = _coerce_query(item)
                if coerced:
                    return coerced
            return None
        if isinstance(value, dict):
            for key in ("query", "question", "prompt", "text", "content"):
                if key in value:
                    coerced = _coerce_query(value[key])
                    if coerced:
                        return coerced
            if "messages" in value:
                coerced = _coerce_query(value["messages"])
                if coerced:
                    return coerced
            if "input" in value and value["input"] is not value:
                coerced = _coerce_query(value["input"])
                if coerced:
                    return coerced
            for val in value.values():
                coerced = _coerce_query(val)
                if coerced:
                    return coerced
            return None
        if isinstance(value, list):
            for item in reversed(value):
                coerced = _coerce_query(item)
                if coerced:
                    return coerced
            return None
        return None

    q = _coerce_query(state.get("query"))
    if not q:
        q = _coerce_query(state.get("input"))
    if not q:
        q = _coerce_query({k: v for k, v in state.items() if k not in {"hits", "rewritten", "answer"}})
    if not q:
        raise ValueError("Missing 'query' in agent state")

    state["query"] = q

    hits = state.get("hits")
    if not isinstance(hits, list):
        incoming = state.get("input")
        if isinstance(incoming, dict):
            hits = incoming.get("hits")
        if not isinstance(hits, list):
            hits = []
    state["hits"] = hits

    return q


def node_analyze(state: AgentState):
    q = _hydrate_state(state)
    sys = "Classify user intent (qa/list/count) and propose a short rewritten query for retrieval. Return as JSON with fields: intent, rewritten."
    msg = adapter._chat.invoke([("system", sys), ("user", q)])
    state.update({"rewritten": msg.content})
    return state

# retrieval is done by Server1 via HTTP; here we assume injected results in state["hits"]

def node_generate(state: AgentState):
    q = _hydrate_state(state)
    context = "\n\n".join([h["text"] for h in state.get("hits", [])])
    sys = "You are a strict RAG assistant. Answer ONLY from Context. If missing, say 'Không tìm thấy trong tài liệu'. For list/count, double-check numbers from context."
    user = f"Context:\n{context}\n\nQuestion: {q}"
    msg = adapter._chat.invoke([("system", sys), ("user", user)])
    state["answer"] = msg.content
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", node_analyze)
    g.add_node("generate", node_generate)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "generate")
    g.add_edge("generate", END)
    return g.compile()