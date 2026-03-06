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
- **Model selection**: UNet, Attention UNet, SegResNet, SwinUNETR, BasicUNet, BasicUNet++, DynUNet, VNet, HighResNet, UNETR, SegResNetVAE, SegResNetDS/DS2, FlexibleUNet, DiNTS, MedNeXt (S/M/B/L)
- **Dataset selection**: Auto-discovery from nnUNet format
- **Training configuration**: Epochs, batch size, learning rate, optimizer, LR scheduler
- **Dataset classes**: Dataset, CacheDataset, PersistentDataset, SmartCacheDataset, LMDBDataset, CacheNTransDataset, ArrayDataset, ZipDataset, GridPatchDataset, PatchDataset, DecathlonDataset — with independent train/inference settings (cache rate, replace rate, cache dir)
- **Preprocessing**: MinMax normalization, Z-score normalization, center/random cropping
- **Augmentation**: Unified tab with basic (Rotate, Flip), spatial transforms (RandAffine, Rand2DElastic, RandRotate90, etc.), and intensity transforms (RandGibbsNoise, RandBiasField, etc.) — probability sliders on all random transforms
- **Metrics**: Dice, IoU, Hausdorff Distance, Surface Distance, Surface Dice (NSD), Generalized Dice, Confusion Matrix (F1), F-Beta Score, Panoptic Quality
- **Loss functions**: Dice, Cross Entropy, Focal, Dice+CE, Dice+Focal, Generalized Dice, Tversky, Hausdorff DT, SSIM, and more
- **Deep supervision**: Toggle that wraps any loss with multi-scale supervision (DynUNet, SegResNetDS, MedNeXt)
- **Inferer**: Simple, Sliding Window, Patch, Saliency (GradCAM), Slice (2D on 3D)
- **Device**: GPU (CUDA/BF16/FP16) or CPU with mixed precision support
- **2D and 3D** image support with automated scaling via PyTorch Lightning Fabric

### Configs Tab
- **Save and manage** training configurations from the Generate tab
- **Config cards** with launch-bar UI (summary, progress bar, Launch/Stop buttons) and color-coded borders (red=idle, yellow=training done, green=inference done)
- **K-fold cross-validation** — each config auto-generates fold variants (No Val, Fold 1–K) with per-fold state tracking (status, run directory, checkpoints, W&B run)
- **Concurrent training** — launch multiple configs and folds in parallel, each with independent progress tracking
- **Inference mode** — Infer button on completed configs runs `--mode infer` to evaluate the test set with Dice/IoU metrics
- **Per-card log streaming** via Server-Sent Events (SSE) with expandable terminal view
- **Re-launch** — completed configs can be re-launched to start fresh training; interrupted runs auto-resume from last checkpoint
- **Auto-reattach** running processes on page reload
- **Stop training** with graceful shutdown (terminate → kill fallback)
- **Sync W&B** — synchronize local configs with Weights & Biases (deletes orphaned runs, updates metadata)

### Weights & Biases
- **Automatic logging** to W&B project `AutoMONAI` during training (loss, Dice, IoU per epoch)
- **Inference metrics** logged with `infer/` prefix (e.g. `infer/dice`, `infer/iou`) to both charts and run summary
- **Unified runs** — training and inference log to the same W&B run automatically
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
ruff check automonai/              # Python
ruff format automonai/
biome check ./automonai/frontend/  # JavaScript
biome check --write ./automonai/frontend/

# Test
uv run pytest automonai/core/tests/ automonai/backend/tests/
```

### Code Organization

See [AGENTS.md](./AGENTS.md) for detailed coding guidelines:
- **Python style**: Snake case, f-strings, line length 100
- **JavaScript modules**: 8 focused modules in `automonai/frontend/js/` (API, command building, configs, UI actions, navigation, search, theme, initialization)
- **CSS modules**: 6 focused stylesheets in `automonai/frontend/styles/` (base, components, augmentation, modals, launch, configs)

## Project Structure

```
.
├── automonai/                    # Main package
│   ├── core/                    # Core ML library
│   │   ├── models.py           # Model factory
│   │   ├── dataset.py          # Dataset utilities
│   │   ├── transforms.py       # Data transforms
│   │   ├── inference.py        # Inference utilities
│   │   ├── cli.py              # CLI tools
│   │   ├── config.py           # Configuration
│   │   ├── results.py          # RunLogger (checkpoints, metrics CSV, summaries)
│   │   └── tests/              # Comprehensive test suite
│   ├── backend/                 # FastAPI server
│   │   ├── server.py           # FastAPI app factory
│   │   ├── routers/            # Modularized API routes
│   │   │   ├── config.py      # Models & datasets endpoints
│   │   │   ├── configs.py     # Config CRUD & W&B sync
│   │   │   └── launch.py      # Training/inference launch & streaming logs
│   │   └── cli/                # CLI entry points
│   │       └── gui.py         # Web UI launcher
│   └── frontend/                # Web UI (HTML, CSS, JS)
│       ├── index.html          # Main HTML
│       ├── js/                 # Modularized JavaScript (8 modules)
│       │   ├── api.js, command.js, configs.js, ui-actions.js
│       │   ├── theme.js, nav.js, search.js, init.js
│       ├── styles/             # Modularized CSS (6 stylesheets)
│       │   ├── base.css, components.css, augmentation.css
│       │   ├── modals.css, launch.css, configs.css
│       └── launch.js           # Legacy stub (syncLaunchCommand)
├── results/                     # Training run outputs
│   └── {dataset}/{model}/{timestamp}/  # Config, metrics, checkpoints per run
├── pyproject.toml               # Project config & dependencies
├── AGENTS.md                    # Coding guidelines for agents
├── LICENSE                      # MIT License
└── .pre-commit-config.yaml      # Pre-commit hooks (prek)
```

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
