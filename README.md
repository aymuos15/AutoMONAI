# AutoMONAI

Medical image segmentation library with web and terminal UIs.

## Quick Start

```bash
# Install
uv pip install -e .

# Launch Web UI (opens in browser)
automonai-gui

# Launch Terminal UI (in another terminal)
automonai-tui
```

The server runs on `http://localhost:8888` with API docs at `/docs`.

## API

- Model selection (UNet, Attention UNet, SegResNet, SwinUNETR)
- Dataset selection with auto-discovery from nnUNet format
- Training configuration (epochs, batch size, learning rate, etc.)
- Dataset classes (Dataset, CacheDataset, PersistentDataset, SmartCacheDataset)
- Augmentation selection
- 2D and 3D image support
- GPU/CPU device selection with automated scaling via lightning fabric

## Development

```bash
# Install dev tools
uv pip install -e ".[dev]"
prek install

# Lint & format
ruff check src/              # Python
ruff format src/
go vet ./UI/tui/...         # Go
gofumpt -w ./UI/tui/
biome check ./UI/gui/       # JavaScript

# Test
uv run pytest src/tests/
```

## Project Structure

```
.
├── src/                      # Core library
│   ├── models.py            # Model factory
│   ├── dataset.py           # Dataset utilities
│   ├── transforms.py        # Data transforms
│   ├── inference.py         # Inference utilities
│   ├── cli.py               # CLI tools
│   ├── config.py            # Configuration
│   └── tests/               # Comprehensive test suite
├── UI/                       # User interfaces
│   ├── server.py            # FastAPI backend (serves both UIs)
│   ├── cli/                 # CLI entry points
│   │   ├── gui.py          # Web UI launcher
│   │   └── tui.py          # Terminal UI launcher
│   ├── gui/                 # Web UI (HTML, CSS, JS)
│   └── tui/                 # Terminal UI (Go + Charmbracelet)
├── predictions/             # Output directory for inference results
├── pyproject.toml            # Project config & dependencies
└── .pre-commit-config.yaml   # Pre-commit hooks (prek)
```
