````markdown
# Agentic RAG (3 servers)

### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && edit .env
````

### 1) Server 1 — Retrieval

* **POST /ingest** `{folder, reset}` → preprocess (OCR if page text too short) + build FAISS index.
* **POST /search** `{query, top_k, rewrite, use_multi_query, use_compression, ...}` → retrieve chunks.

```bash
uvicorn retrieval.server:app --reload --port 8001
```

### 2) Server 2 — RAG (LangGraph Agent)

* **POST /ask** `{query}` → runs small graph; calls server1 for retrieval; returns `{answer, citations, locations}` with bbox quads (first pages) and can optionally generate a `*_highlighted.pdf` using `rag/locator.py`.

```bash
uvicorn rag.server:app --reload --port 8002
```

### 3) Server 3 — UI (Chat + PDF preview)

```bash
uvicorn ui.server:app --reload --port 8003
open http://localhost:8003
```

### Notes

* Default **Gemini** for LLM + Embedding via `config/models.py`. Swap to OpenAI/Azure by extending `LLMAdapter`.
* Bounding boxes: page-index uses **PyMuPDF words** when text layer is good; otherwise **PaddleOCR** to get boxes.
* This repo intentionally keeps the RAG graph simple; extend nodes in `rag/graph.py` for: re-rank, answer critique, tool-use for counting/listing.

```
```
