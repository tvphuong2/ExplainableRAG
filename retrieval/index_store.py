from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings as C
from config.models import LLMConfig, LLMAdapter


def build_embeddings():
    adapter = LLMAdapter(LLMConfig(C.PROVIDER, C.GENERATION_MODEL, C.EMBED_MODEL))
    return adapter


def ingest_folder(prep_dir: Path, vs_dir: Path):
    adapter = build_embeddings()
    text_dir = Path(prep_dir)/"text"
    files = sorted(text_dir.glob("*.txt"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, add_start_index=True)
    docs: List[Document] = []
    for p in files:
        content = p.read_text(encoding='utf-8')
        for i, chunk, start in splitter.create_documents_returning_splits([content]):
            # helper not available -> fallback manual
            pass