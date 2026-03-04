# AGENTS.md

Guidelines for agentic coding agents (e.g. Claude, Copilot, Cursor) working in this repository.

---

## Project Overview

**AutoMONAI** is a Python medical image segmentation framework built on MONAI + PyTorch Lightning Fabric.
It exposes two interfaces: a core Python CLI (`src/run.py`) and a web GUI (`UI/gui/`). A FastAPI server
(`UI/server.py`, port 8888) bridges the GUI to the core library.

Languages: **Python** (core library, FastAPI server, tests), **HTML/CSS/JS** (web GUI).
Training metrics are logged to **Weights & Biases** (project name: `AutoMONAI`).

---

## Repository Layout

```
src/            Core Python library (config, models, dataset, transforms, train, inference, results, cli, run)
  results.py    RunLogger — checkpoints, metrics CSV, config/summary JSON per run
  tests/        pytest test suite (test_models, test_losses, test_integration)
UI/
  server.py     FastAPI app factory (~51 lines) — mounts routers, serves index.html, /static
  routers/      Modularized API routes
    config.py   GET /api/models, /api/datasets
    launch.py   POST /api/launch, GET /api/launch/{status,logs,list}, POST /api/launch/stop
    results.py  GET /api/results, DELETE /api/results/{dataset}/{model}/{timestamp}
    configs.py  GET/POST/DELETE /api/configs, POST /api/configs/sync-wandb
  cli/          CLI entry points (gui.py)
  gui/          Web interface (vanilla HTML/CSS/JS, no build step)
    index.html  HTML structure
    js/         Modularized JavaScript (7 modules)
      api.js, command.js, configs.js, ui-actions.js, theme.js, nav.js, search.js, init.js
    styles/     Modularized CSS (7 modules)
      base.css, components.css, augmentation.css, modals.css, results.css, launch.css, configs.css
    launch.js   Launch page UI — passes run_id "__main__" for single-run launches
    results.js  Results viewer with Chart.js graphs (unchanged)
results/        Training run outputs (results/dataset/model/timestamp/{config,metrics,summary,checkpoints})
pyproject.toml  Build config, dependencies, ruff, pytest, coverage settings
```

---

## Build / Run Commands

```bash
uv sync                        # install all deps from uv.lock
uv pip install -e ".[dev]"     # editable install with dev extras
automonai-gui                  # launch web GUI (FastAPI server + browser)
automonai-train --dataset Dataset001_Cellpose --model unet --epochs 10
```

---

## Test Commands

```bash
pytest                                                         # full suite
pytest src/tests/test_models.py                                # single file
pytest src/tests/test_models.py::TestModelCreation             # single class
pytest src/tests/test_models.py::TestModelCreation::test_unet_creation  # single test
pytest -k "unet and not slow"                                  # keyword filter
pytest -m unit                                                 # by marker
pytest -m "not cuda and not slow"
pytest --cov=src --cov-report=term-missing                     # with coverage
pytest -n auto                                                 # parallel (pytest-xdist)
```

Available markers (declare with `@pytest.mark.<name>`):
- `cuda` — requires CUDA GPU
- `slow` — long-running tests
- `integration` — end-to-end tests
- `unit` — isolated unit tests

---

## Lint / Format Commands

```bash
ruff check src/                # lint Python (check only)
ruff check --fix src/          # lint with auto-fix
ruff format src/               # format Python
ty check src/                  # type-check (ty, not mypy)
prek run --all-files           # all pre-commit hooks
prek install                   # install hooks
```

### UI-Specific

```bash
# JavaScript (GUI)
biome check UI/gui/            # lint JS (check only)
biome check --write UI/gui/    # lint + format JS
biome format --write UI/gui/   # format JS only
```

Install UI lint tools:

```bash
# JavaScript linter - Linux
curl -L https://github.com/biomejs/biome/releases/latest/download/biome-linux-x64 -o ~/.local/bin/biome
chmod +x ~/.local/bin/biome
```

---

## Python Code Style

**Imports** — order: stdlib → third-party → local (`.config`, `.models`). One blank line between
groups, none within. Use lazy imports inside functions only for optional heavy deps (e.g. `nibabel`
only in 3D code paths).

**Formatting** — line length 100 (ruff), double quotes, 4-space indent, f-strings only (no `%` or
`.format()`).

**Types** — `src/` intentionally omits annotations to stay minimal; add them when introducing new
functions or when `ty` flags an error. Use `# type: ignore[attr-defined]` with the error code when
suppression is unavoidable. Guard import-only references with `TYPE_CHECKING`.

**Naming**

| Construct | Style | Example |
|---|---|---|
| Functions / variables | `snake_case` | `get_model`, `train_one_epoch` |
| Classes | `PascalCase` | `TrainDataset`, `DictTransform` |
| Module-level constants | `UPPER_SNAKE_CASE` | `DATASET_ROOT`, `TRAINING_DEFAULTS` |
| Private helpers | `_leading_underscore` | `_load_json` |
| Test classes | `TestPascalCase` | `TestModelCreation` |

**Error handling** — raise `ValueError` with a descriptive f-string for bad user input:

```python
raise ValueError(f"Unknown model: {model_name}")
```

For validation errors, include available options when helpful:

```python
raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
```

`except Exception: continue` is acceptable only in config-discovery loops where a bad file should
be silently skipped. Avoid bare `except` elsewhere.

**Docstrings** — required (one-line) for all FastAPI route handlers in `UI/server.py` and `UI/routers/*.py`;
optional but welcome for public `src/` functions; required (one-line) for test classes.

---

## Test Style

Class-based pytest — one class per logical unit, `device` fixture defined locally in each class.

```python
class TestFoo:
    """Tests for the foo subsystem."""

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet"])
    def test_forward_pass(self, model_name):
        device = torch.device("cpu")
        model = get_model(model_name, in_channels=1, out_channels=2).to(device)
        out = model(torch.randn(1, 1, 64, 64, device=device))
        assert out.shape == (1, 2, 64, 64)
        assert not torch.isnan(out).any()
```

- Use dummy tensor data (`torch.randn`, `torch.randint`) — never load real files in unit tests
- `pytest.mark.parametrize` for model × loss × dimensionality matrices
- `pytest.raises(ValueError, match="...")` for error-path testing
- `@pytest.mark.skip(reason="...")` or `@pytest.mark.slow` for memory-prohibitive tests

---

## JavaScript (GUI) Style

- Vanilla ES2020, no build step, no npm
- `const` by default, `let` when reassignment is needed, never `var`
- Fetch API for all HTTP calls; handle errors with `.catch()`
- DOM updates via `textContent` / `value` — avoid `innerHTML` with user data

### GUI Module Organization (Feb 27, 2026)

`UI/gui/` is modularized into 8 focused JS files + 7 CSS files:

**JavaScript modules** (`UI/gui/js/`):
- `api.js` — data loading (datasets, models, classes)
- `command.js` — command building & formatting (main bulk)
- `configs.js` — config card rendering, per-card launch/stop/progress, SSE log streams, W&B sync
- `ui-actions.js` — copy to clipboard, modal open/close
- `theme.js` — dark/light theme toggle
- `nav.js` — page & sub-tab navigation
- `search.js` — tab search modal logic
- `init.js` — global keydown listeners, DOMContentLoaded setup

Load order in `index.html`: `theme` → `ui-actions` → `api` → `command` → `configs` → `nav` → `search` → `results.js` → `launch.js` → `init.js`

**CSS modules** (`UI/gui/styles/`):
- `base.css` — CSS variables, resets, forms (must load first; defines `--bg`, `--fg`, etc.)
- `components.css` — buttons, badges, nav tabs, page/sub-page show/hide
- `augmentation.css` — augmentation controls, metrics checkboxes, doc sections
- `modals.css` — tab search modal, command modal styles
- `results.css` — results page all styles
- `launch.css` — launch page all styles
- `configs.css` — config card spacing and terminal log styling

Load order: `base.css` first (defines vars), then others in any order (no cascade dependencies)
