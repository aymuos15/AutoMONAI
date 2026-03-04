# AutoMONAI

Medical image segmentation library with a web UI for interactive training and inference.

## Quick Start

```bash
# Install
uv pip install -e .

# Launch Web UI (opens in browser)
automonai-gui
```

The server runs on `http://localhost:8888` with API docs at `/docs`.

## Features

### Generate Tab
- **Model selection**: UNet, Attention UNet, SegResNet, SwinUNETR
- **Dataset selection**: Auto-discovery from nnUNet format
- **Training configuration**: Epochs, batch size, learning rate, optimizer, LR scheduler
- **Dataset classes**: Dataset, CacheDataset, PersistentDataset, SmartCacheDataset
- **Preprocessing**: MinMax normalization, Z-score normalization, center/random cropping
- **Augmentation**: Rotation and flip transforms with probability control
- **Metrics**: Dice and IoU (Intersection over Union)
- **Loss functions**: Dice, Cross Entropy, Focal Loss
- **Device**: GPU (CUDA/BF16/FP16) or CPU with mixed precision support
- **2D and 3D** image support with automated scaling via PyTorch Lightning Fabric

### Configs Tab
- **Save and manage** training configurations from the Generate tab
- **Config cards** with launch-bar UI (summary, progress bar, Launch/Stop buttons)
- **Concurrent training** — launch multiple configs in parallel, each with independent progress tracking
- **Inference mode** — Infer button on completed configs runs `--mode infer` to evaluate the test set with Dice/IoU metrics
- **Per-card log streaming** via Server-Sent Events (SSE) with expandable terminal view
- **Auto-reattach** running processes on page reload
- **Stop training** with graceful shutdown (terminate → kill fallback)
- **Sync W&B** — synchronize local configs with Weights & Biases (deletes orphaned runs, updates metadata)

### Weights & Biases
- **Automatic logging** to W&B project `AutoMONAI` during training (loss, Dice, IoU per epoch)
- **Inference metrics** logged with `infer/` prefix (e.g. `infer/dice`, `infer/iou`) to both charts and run summary
- **Run resume** support via `--run_id` for interrupted training sessions
- **Sync button** in Configs tab cleans up orphaned W&B runs and updates run metadata

### Additional
- **Keyboard shortcuts**: `Ctrl+Shift+H` to view all shortcuts
- **Dark/Light theme** with localStorage persistence
- **Responsive design** for desktop browsing

## Development

```bash
# Install dev tools
uv pip install -e ".[dev]"
prek install

# Lint & format
ruff check src/              # Python
ruff format src/
biome check ./UI/gui/       # JavaScript
biome check --write ./UI/gui/

# Test
uv run pytest src/tests/
```

### Code Organization

See [AGENTS.md](./AGENTS.md) for detailed coding guidelines:
- **Python style**: Snake case, f-strings, line length 100
- **JavaScript modules**: 8 focused modules in `UI/gui/js/` (API, command building, configs, UI actions, navigation, search, theme, initialization)
- **CSS modules**: 6 focused stylesheets in `UI/gui/styles/` (base, components, augmentation, modals, launch, configs)

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
│   ├── results.py           # RunLogger (checkpoints, metrics CSV, summaries)
│   └── tests/               # Comprehensive test suite
├── UI/                       # Web User Interface
│   ├── server.py            # FastAPI app factory (51 lines)
│   ├── routers/             # Modularized API routes
│   │   ├── config.py       # Models & datasets endpoints
│   │   ├── configs.py      # Config CRUD & W&B sync
│   │   └── launch.py       # Training/inference launch & streaming logs
│   ├── cli/                 # CLI entry points
│   │   └── gui.py          # Web UI launcher
│   └── gui/                 # Web UI (HTML, CSS, JS)
│       ├── index.html      # Main HTML
│       ├── js/             # Modularized JavaScript (8 modules)
│       │   ├── api.js, command.js, configs.js, ui-actions.js
│       │   ├── theme.js, nav.js, search.js, init.js
│       ├── styles/         # Modularized CSS (6 stylesheets)
│       │   ├── base.css, components.css, augmentation.css
│       │   ├── modals.css, launch.css, configs.css
│       └── launch.js       # Legacy stub (syncLaunchCommand)
├── results/                 # Training run outputs
│   └── {dataset}/{model}/{timestamp}/  # Config, metrics, checkpoints per run
├── pyproject.toml           # Project config & dependencies
├── AGENTS.md                # Coding guidelines for agents
├── LICENSE                  # MIT License
└── .pre-commit-config.yaml  # Pre-commit hooks (prek)
```

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
