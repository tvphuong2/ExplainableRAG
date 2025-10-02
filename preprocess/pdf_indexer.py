from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import fitz
from PIL import Image
from shared.utils import normalize_text
from config import settings as C
from .ocr import ocr_image_with_boxes

# Build per-page index with (text + word-level spans with bbox)

def build_page_index(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    pages = []
    try:
        raw_pages = []
        for p in doc:
            raw_pages.append(p.get_text("text") or "")
        # heuristic: if a page text too short -> OCR fallback
        for i, page in enumerate(doc):
            txt = normalize_text(raw_pages[i])
            page_w, page_h = page.rect.width, page.rect.height
            if len(txt) < C.MIN_TEXT_LEN:
                # render and OCR with boxes
                mat = fitz.Matrix(C.OCR_DPI/72, C.OCR_DPI/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr = ocr_image_with_boxes(img)
                spans = []
                cur = 0
                buf = []
                for (x0,y0,x1,y1), t in ocr:
                    t2 = t.strip()
                    if not t2: continue
                    if buf and not buf[-1].endswith(' '):
                        buf.append(' '); cur += 1
                    s = cur; buf.append(t2); cur += len(t2)
                    spans.append({"start": s, "end": cur, "text": t2, "bbox": [float(x0),float(y0),float(x1),float(y1)]})
                pages.append({
                    "file": str(pdf_path), "page": i+1, "width": page_w, "height": page_h,
                    "mode": "pdf_ocr", "text": normalize_text(''.join(buf)), "spans": spans
                })
            else:
                words = page.get_text("words", sort=True)
                spans = []
                cur = 0; buf = []
                for (x0,y0,x1,y1,w,*_) in words:
                    if buf and not buf[-1].endswith(' '):
                        buf.append(' '); cur += 1
                    s = cur; buf.append(w); cur += len(w)
                    spans.append({"start": s, "end": cur, "text": w, "bbox": [float(x0),float(y0),float(x1),float(y1)]})
                pages.append({
                    "file": str(pdf_path), "page": i+1, "width": page_w, "height": page_h,
                    "mode": "pdf_text", "text": normalize_text(''.join(buf)), "spans": spans
                })
        return pages
    finally:
        doc.close()