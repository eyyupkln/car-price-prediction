from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.services.predictor import get_model, get_options


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"

app = FastAPI(
    title="Car Price Prediction API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api", tags=["api"])


@app.on_event("startup")
def _startup() -> None:

    get_model()
    get_options()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)
