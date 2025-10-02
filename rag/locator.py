from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json
from rapidfuzz import fuzz, process
import fitz

# Map snippet text -> quads using page_index jsonl

def find_quads(pdf_path: Path, page_index_path: Path, snippet: str, score_cutoff: int = 85):
    # read index into memory (small per file)
    pages = []
    with open(page_index_path, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))
    hits = {}
    for p in pages:
        text = p["text"]
        # very simple fuzzy gating
        if process.extractOne(snippet, [text], scorer=fuzz.partial_ratio, score_cutoff=score_cutoff):
            # collect word boxes overlapping the best exact location if any
            lo = text.lower().find(snippet.lower())
            hi = lo + len(snippet) if lo >= 0 else None
            quads = []
            for s in p["spans"]:
                if hi is None or (s["end"] <= lo or s["start"] >= hi):
                    continue
                x0,y0,x1,y1 = s["bbox"]
                quads.append(fitz.Quad(fitz.Rect(x0,y0,x1,y1))).to_dict()
            if quads:
                hits[p["page"]] = quads
    return hits


def write_highlighted(pdf_path: Path, quads_map: dict, out_path: Path):
    doc = fitz.open(str(pdf_path))
    try:
        for page_no, quads in quads_map.items():
            page = doc[page_no-1]
            page.add_highlight_annot([fitz.Quad(q) for q in quads])
        doc.save(str(out_path))
    finally:
        doc.close()