from __future__ import annotations
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from config.models import LLMAdapter, LLMConfig
from config import settings as C

adapter = LLMAdapter(LLMConfig(C.PROVIDER, C.GENERATION_MODEL, C.EMBED_MODEL))

class AgentState(dict):
    pass

def _hydrate_state(state: AgentState) -> str:
    """Ensure required fields are present regardless of langgraph input format."""
    if "query" not in state or "hits" not in state:
        incoming = state.get("input")
        if isinstance(incoming, dict):
            state.setdefault("query", incoming.get("query"))
            state.setdefault("hits", incoming.get("hits", []))

    q = state.get("query")
    if q is None:
        raise ValueError("Missing 'query' in agent state")

    state.setdefault("hits", [])
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