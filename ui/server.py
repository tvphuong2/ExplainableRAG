from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, requests
from pathlib import Path
import fitz

from config import settings as C

app = FastAPI(title="Server3 - UI", version="1.0")
app.mount('/static', StaticFiles(directory=Path(__file__).parent/'static'), name='static')
tpl = Jinja2Templates(directory=str(Path(__file__).parent/'templates'))

RAG_URL = os.getenv('RAG_URL', 'http://localhost:8002')

class Ask(BaseModel):
    query: str

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return tpl.TemplateResponse('chat.html', {"request": request})

@app.post('/ask')
async def ask(req: Ask):
    r = requests.post(f"{RAG_URL}/ask", json={"query": req.query})
    return r.json()

@app.get('/view')
async def view(file: str, page: int = 1):
    # simple preview render as PNG with existing highlights (if any)
    pdf = fitz.open(file)
    p = pdf[page-1]
    pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
    out = Path(C.DATA_DIR)/'preview.png'
    pix.save(out)
    pdf.close()
    html = f"""
    <html><body>
    <div><img src='file://{out.resolve()}' style='max-width:90%;border:1px solid #ddd;border-radius:8px;'/></div>
    <p><i>Nếu chưa thấy highlight, hãy dùng viewer PDF mặc định để xem file đã chèn annotation.</i></p>
    </body></html>
    """
    return HTMLResponse(html)