from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil, json
from typing import Optional, List, Dict, Any

from config import settings as C
from config.models import LLMConfig, LLMAdapter
from preprocess.pipeline import preprocess_folder

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

app = FastAPI(title="Server1 - Retrieval", version="1.0")
VS_PATH = Path(C.INDEX_DIR)/"faiss_index"
adapter = LLMAdapter(LLMConfig(C.PROVIDER, C.GENERATION_MODEL, C.EMBED_MODEL))

class IngestReq(BaseModel):
    folder: str
    reset: bool = True

class SearchReq(BaseModel):
    query: str
    top_k: int = C.TOP_K
    rewrite: bool = True
    use_multi_query: bool = True
    use_compression: bool = False
    search_type: str = "mmr"  # or similarity
    mmr_lambda: float = C.MMR_LAMBDA

@app.post("/ingest")
def ingest(req: IngestReq):
    src = Path(req.folder)
    if not src.exists():
        raise HTTPException(400, "folder not found")
    # preprocess → prepped/text + page_index
    preprocess_folder(src, C.PREP_DIR)
    # build FAISS
    if req.reset and VS_PATH.exists():
        shutil.rmtree(VS_PATH, ignore_errors=True)
    text_dir = Path(C.PREP_DIR)/"text"
    files = sorted(text_dir.glob("*.txt"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, add_start_index=True)
    docs: List[Document] = []
    for p in files:
        content = p.read_text(encoding='utf-8')
        splits = splitter.split_text(content)
        for i, chunk in enumerate(splits):
            docs.append(Document(page_content=chunk, metadata={"file_name": p.stem, "start_index": getattr(splitter, 'last_start_index', 0)}))
    vs = FAISS.from_documents(docs, adapter._emb)
    VS_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VS_PATH))
    return {"ok": True, "num_chunks": len(docs)}


def _load_vs():
    if not VS_PATH.exists():
        return None
    return FAISS.load_local(str(VS_PATH), adapter._emb, allow_dangerous_deserialization=True)

class Hit(BaseModel):
    file_name: str
    text: str

class SearchResp(BaseModel):
    rewritten: Optional[str]
    hits: List[Hit]

@app.post("/search", response_model=SearchResp)
def search(req: SearchReq):
    vs = _load_vs()
    if vs is None:
        raise HTTPException(400, "index not ready")

    retriever = vs.as_retriever(search_type=req.search_type, search_kwargs={"k": req.top_k, "lambda_mult": req.mmr_lambda})
    llm = adapter._chat

    if req.rewrite:
        sys = "Rewrite the query briefly for retrieval; keep language and key terms."
        newq = llm.invoke([("system", sys), ("user", f"Original: {req.query}\nOne line rewritten:")]).content
    else:
        newq = req.query

    if req.use_multi_query:
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True)
    if req.use_compression:
        retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=LLMChainExtractor.from_llm(llm))

    docs = retriever.get_relevant_documents(newq)
    hits = [Hit(file_name=d.metadata.get("file_name"), text=d.page_content) for d in docs]
    return SearchResp(rewritten=newq if req.rewrite else None, hits=hits)