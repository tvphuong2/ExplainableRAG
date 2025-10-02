from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os
from config.settings import *

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

@dataclass
class LLMConfig:
    provider: str
    chat_model: str
    embed_model: str

class LLMAdapter:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.provider = cfg.provider
        if self.provider == "gemini":
            key = GEMINI_API_KEY
            if not key:
                raise RuntimeError("Missing GEMINI_API_KEY")
            self._chat = ChatGoogleGenerativeAI(model=cfg.chat_model, google_api_key=key, temperature=0.0)
            self._emb = GoogleGenerativeAIEmbeddings(model=cfg.embed_model, google_api_key=key, task_type="retrieval_document", batch_size=96)
        else:
            raise NotImplementedError("Only Gemini adapter is wired. Add OpenAI/Azure later using same interface.")

    # Chat / completion
    def invoke(self, messages: List[tuple]) -> str:
        resp = self._chat.invoke(messages)
        return resp.content or ""

    # Embeddings
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)