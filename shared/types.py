from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PageSpan:
    start: int
    end: int
    text: str
    bbox: list  # [x0,y0,x1,y1]

@dataclass
class PageIndex:
    file: str
    page: int
    width: int
    height: int
    mode: str     # pdf_text | pdf_ocr
    text: str     # page text normalized
    spans: List[PageSpan]

@dataclass
class CitationLoc:
    file: str
    page: int
    quads: list   # list of 8-float quads or rects

@dataclass
class AskOutput:
    answer: str
    citations: List[Dict[str, Any]]
    locations: List[CitationLoc]