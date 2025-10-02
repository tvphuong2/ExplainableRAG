from __future__ import annotations
import re, hashlib
from ftfy import fix_text

BULLETS = {"\u2022": "- ", "\u2023": "- ", "\u25E6": "- ", "\u2219": "- ", "\u25CF": "- ", "\u25A0": "- "}

def normalize_text(s: str) -> str:
    s = fix_text(s or "")
    for k,v in BULLETS.items():
        s = s.replace(k,v)
    s = re.sub(r"-\n(\w)", r"\1", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*\n\s*", "\n\n", s)
    return s.strip()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()