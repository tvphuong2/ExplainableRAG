from __future__ import annotations
from typing import List, Tuple
from PIL import Image
from paddleocr import PaddleOCR

_ocr_singleton = None

def _get_paddle_ocr():
    global _ocr_singleton
    if _ocr_singleton is None:
        # enable angle cls and multilingual model
        _ocr_singleton = PaddleOCR(use_angle_cls=True, lang='vi')
    return _ocr_singleton

# return [(bbox, text)] where bbox is [x0,y0,x1,y1]
# Paddle returns polygons; we convert to min-rect

def ocr_image_with_boxes(img: Image.Image) -> List[Tuple[list, str]]:
    ocr = _get_paddle_ocr()
    import numpy as np
    arr = np.asarray(img)
    res = ocr.ocr(arr, cls=True)
    out = []
    for line in res[0]:
        poly = line[0]
        txt = line[1][0]
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        out.append((bbox, txt))
    return out