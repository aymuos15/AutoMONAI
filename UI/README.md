# MonaiUI Frontend

This directory contains the complete UI stack for MonaiUI:
- **Web UI (GUI)**: Browser-based interface
- **Terminal UI (TUI)**: Command-line interface built with Go
- **Shared API**: FastAPI backend serving both UIs

## Structure

```
UI/
├── server.py          # FastAPI server (serves both web UI and API)
├── gui/               # Web interface (HTML, CSS, JS)
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── tui/               # Terminal interface (Go)
│   ├── main.go
│   ├── go.mod
│   ├── go.sum (auto-generated)
│   └── monaiui-tui (compiled binary)
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+ (for FastAPI server)
- Go 1.21+ (for TUI, optional)
- Node.js/npm (for web UI development, optional)

### Running the Server & Web UI

```bash
# From the project root
python UI/server.py
```

Then open your browser:
- 🌐 Web UI: http://localhost:8888
- 📚 API Docs: http://localhost:8888/docs
- 🔌 OpenAPI Schema: http://localhost:8888/openapi.json

### Running the Terminal UI

Make sure the FastAPI server is already running, then:

```bash
cd UI/tui
go mod download
go build -o monaiui-tui .
./monaiui-tui
```

The TUI will automatically fetch the API schema from the running server.

## Development

### Web UI
Edit files in `gui/`:
- `index.html` - Structure
- `styles.css` - Styling
- `script.js` - Interactions

Changes are served automatically (requires server restart for HTML changes).

### FastAPI Server
Edit `server.py` to:
- Add new API endpoints
- Modify request/response models
- The OpenAPI schema updates automatically

### Terminal UI
See [`tui/README.md`](tui/README.md) for development guide.

## API

Both UIs communicate via the FastAPI server. The API schema is automatically generated and available at `/openapi.json`.

Key endpoints:
- `GET /api/models` - List available models
- `GET /api/datasets` - List available datasets
- `POST /api/train` - Start training (to be implemented)
- `POST /api/infer` - Run inference (to be implemented)

Add more endpoints in `server.py` to expose functionality from `/src/`.

## Deployment

### Web UI
- Build: Already bundled with server
- Deploy: Run `server.py` with a production ASGI server (gunicorn, uvicorn, etc.)

### Terminal UI
- Build: `cd UI/tui && go build -o monaiui-tui .`
- Deploy: Distribute the binary or build on target machine

## Notes

- **Shared Logic**: All real logic stays in `/src/` (Python)
- **API-First**: Both UIs communicate via REST API, making them independent
- **OpenAPI Contract**: Schema at `/openapi.json` is the single source of truth for API definition
