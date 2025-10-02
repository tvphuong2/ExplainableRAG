from __future__ import annotations
from pathlib import Path
import json
from shared.utils import normalize_text, sha256_file
from config import settings as C
from .pdf_indexer import build_page_index

# Consolidate ONE clean .txt per document + page_index JSONL for PDF

def preprocess_folder(raw_dir: Path, out_dir: Path):
    text_dir = Path(out_dir)/"text"; meta_dir = Path(out_dir)/"meta"
    page_dir = Path(C.PAGE_INDEX_DIR)
    text_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in Path(raw_dir).rglob("*") if p.is_file() and p.suffix.lower() in {".pdf",".docx",".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}]
    for src in sorted(files):
        print("[PREP]", src)
        if src.suffix.lower() == ".pdf":
            pages = build_page_index(src)
            # consolidated doc text
            full = normalize_text("\n\n".join([p["text"] for p in pages]))
            (text_dir/f"{src.stem}.txt").write_text(full, encoding='utf-8')
            # page index jsonl
            with (page_dir/f"{src.stem}.jsonl").open('w', encoding='utf-8') as f:
                for p in pages:
                    f.write(json.dumps(p, ensure_ascii=False)+"\n")
            # meta
            meta = {"file": str(src), "sha256": sha256_file(src), "num_pages": len(pages), "length": len(full)}
            (meta_dir/f"{src.stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            # images/docx -> simple: skip for brevity or add docx reader if needed
            pass