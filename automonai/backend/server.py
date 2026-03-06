#!/usr/bin/env python3
"""FastAPI server for MonaiUI web interface and API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from automonai.backend.routers import config, launch, configs

app = FastAPI(
    title="MonaiUI",
    description="Medical image segmentation framework",
    version="0.1.0",
)

# Include routers
app.include_router(config.router)
app.include_router(launch.router)
app.include_router(configs.router)

# Frontend lives at automonai/frontend/ (sibling of backend/)
_frontend_path = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    """Serve index.html."""
    index_path = _frontend_path / "index.html"
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


# Mount static files (CSS, JS, etc.)
if _frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_path)), name="static")


if __name__ == "__main__":
    import uvicorn

    PORT = 8888
    print("Starting MonaiUI...")
    print(f"🌐 Web UI: http://localhost:{PORT}")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"🔌 OpenAPI Schema: http://localhost:{PORT}/openapi.json")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
