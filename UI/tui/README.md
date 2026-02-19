# MonaiUI TUI (Terminal User Interface)

A terminal-based interface for MonaiUI that mirrors the web UI, built with Go and Charmbracelet libraries.

## Prerequisites

- Go 1.21 or later
- The FastAPI server running at `http://localhost:8888`

## Building

```bash
go mod download
go build -o monaiui-tui .
```

## Running

Make sure the FastAPI server is running:
```bash
# In one terminal
python ../server.py
```

Then in another terminal:
```bash
./monaiui-tui
```

## Features

The TUI matches the web UI with two sections:

### Generate Tab
- Select dataset and model
- Choose metrics (Dice, IoU, or both)
- Select loss function (Dice, Cross Entropy, Focal)
- Configure training parameters:
  - Epochs, batch size, learning rate
  - Image size, number of workers
  - Output directory, device (CUDA/CPU)
- Auto-generates training command as you change values
- Copy command for easy terminal execution

### Docs Tab
- Model documentation
- Dataset class descriptions
- Metrics and loss functions info
- Device options reference

## Keyboard Controls

- `Tab` - Switch between Generate and Docs
- `↑/↓` - Navigate form fields
- `Ctrl+C` / `q` - Quit

## Architecture

- **go.mod**: Go dependency management
- **main.go**: TUI application that:
  - Fetches datasets and models from the FastAPI server
  - Displays a form matching the web UI layout
  - Generates commands dynamically
  - Supports two tabs (Generate and Docs)

## Libraries

- **bubbles**: TUI components (textinput, etc.)
- **bubbletea**: Elm-inspired terminal application framework
- **lipgloss**: Terminal styling and layout

## Notes

- The TUI fetches real data from the running FastAPI server
- Generated commands can be directly executed in the terminal
- All functionality matches the web UI exactly
