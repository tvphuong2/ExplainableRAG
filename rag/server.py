from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests, os, json
from pathlib import Path

from config import settings as C
from rag.graph import build_graph
from rag.locator import find_quads

app = FastAPI(title="Server2 - RAG", version="1.0")
GRAPH = build_graph()

SERVER1 = os.getenv("RETRIEVAL_URL", "http://localhost:8001")

class AskReq(BaseModel):
    query: str
    top_k: int = C.TOP_K
    rewrite: bool = True

class AskResp(BaseModel):
    answer: str
    citations: list
    locations: list

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    # 1) call server1 search
    r = requests.post(f"{SERVER1}/search", json={"query": req.query, "top_k": req.top_k, "rewrite": req.rewrite})
    if r.status_code != 200:
        raise HTTPException(400, r.text)
    payload = r.json()
    hits = payload.get("hits", [])

    # 2) run small graph over hits
    st = {"query": req.query, "hits": hits}
    out = GRAPH.invoke(st)
    ans = out.get("answer", "")

    # 3) locate → quads using page index
    locs = []
    for h in hits[:3]:  # limit
        file_stem = h["file_name"]
        page_idx = Path(C.PAGE_INDEX_DIR)/f"{file_stem}.jsonl"
        pdf_path = Path(C.RAW_DIR)/f"{file_stem}.pdf"
        if page_idx.exists() and pdf_path.exists():
            quads_map = find_quads(pdf_path, page_idx, h["text"][:400])
            for page, quads in quads_map.items():
                locs.append({"file": str(pdf_path), "page": page, "quads": quads})

    cits = [{"file": l["file"], "page": l["page"]} for l in locs]
    return AskResp(answer=ans, citations=cits, locations=locs)