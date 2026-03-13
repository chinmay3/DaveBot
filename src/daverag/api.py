from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from daverag.config import settings
from daverag.service import RagService
from daverag.schemas import AskRequest, AskResponse, HealthResponse


app = FastAPI(title=settings.app_name, version="0.1.0")
service = RagService(settings)
frontend_dir = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.on_event("startup")
def startup() -> None:
    service.load_or_build_index()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return service.health()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((frontend_dir / "index.html").read_text())


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return service.ask(question=request.question, topic=request.topic, top_k=request.top_k)
