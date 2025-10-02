from __future__ import annotations
import os
from pathlib import Path


DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR = Path(os.getenv("RAW_DIR", DATA_DIR/"raw"))
PREP_DIR = Path(os.getenv("PREP_DIR", DATA_DIR/"prepped"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", DATA_DIR/"index"))
PAGE_INDEX_DIR = Path(os.getenv("PAGE_INDEX_DIR", DATA_DIR/"page_index"))


# Models
PROVIDER = os.getenv("PROVIDER", "gemini").lower()
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


# OCR
OCR_BACKEND = os.getenv("OCR_BACKEND", "paddle")
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "500"))
TILE_SIZE = int(os.getenv("TILE_SIZE", "2048"))
LARGE_IMG_PIXELS = int(os.getenv("LARGE_IMG_PIXELS", "1500000"))
HEADER_FOOT_FRAC = float(os.getenv("HEADER_FOOT_FRAC", "0.5"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))


# Retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
FETCH_K = int(os.getenv("FETCH_K", "24"))
TOP_K = int(os.getenv("TOP_K", "8"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))


for p in [DATA_DIR, RAW_DIR, PREP_DIR, INDEX_DIR, PAGE_INDEX_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)