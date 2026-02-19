#!/usr/bin/env python3
"""FastAPI server for MonaiUI web interface and API."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import DATASETS, MODELS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="MonaiUI",
    description="Medical image segmentation framework",
    version="0.1.0",
)


@app.get("/api/models")
async def get_models():
    """Get available models."""
    return MODELS


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets."""
    return DATASETS


@app.get("/")
async def root():
    """Serve index.html."""
    index_path = Path(__file__).parent / "gui" / "index.html"
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


# Mount static files (CSS, JS, etc.)
gui_path = Path(__file__).parent / "gui"
if gui_path.exists():
    app.mount("/static", StaticFiles(directory=str(gui_path)), name="static")


if __name__ == "__main__":
    import uvicorn

    PORT = 8888
    print("Starting MonaiUI...")
    print(f"🌐 Web UI: http://localhost:{PORT}")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"🔌 OpenAPI Schema: http://localhost:{PORT}/openapi.json")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
