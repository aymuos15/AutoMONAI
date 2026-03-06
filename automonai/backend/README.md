# AutoMONAI Backend

FastAPI server that bridges the web frontend to the core ML library.

## Structure

```
backend/
├── server.py          # FastAPI app factory — mounts routers, serves frontend
├── routers/           # Modularized API routes
│   ├── config.py     # GET /api/models, /api/datasets, /api/options
│   ├── configs.py    # Config CRUD & W&B sync
│   └── launch.py     # Training/inference launch & streaming logs
├── cli/              # CLI entry points
│   └── gui.py        # Web UI launcher (automonai-gui)
└── configs/          # Saved training configs (JSON)
```

## Quick Start

```bash
# From the project root
automonai-gui                          # CLI entry point (opens browser)
python -m automonai.backend.server     # direct invocation
```

- Web UI: http://localhost:8888
- API Docs: http://localhost:8888/docs

## API

Key endpoints:
- `GET /api/models` - List available models
- `GET /api/datasets` - List available datasets
- `GET /api/options` - All config options (losses, metrics, optimizers, etc.)
- `POST /api/launch` - Launch training
- `GET /api/launch/logs` - Stream logs via SSE
- `POST /api/configs/save` - Save a config
- `GET /api/configs/list` - List all configs

## Notes

- **Shared Logic**: All ML logic lives in `automonai/core/`
- **API-First**: Frontend communicates via REST API
- **OpenAPI Contract**: Schema at `/openapi.json` is the single source of truth
