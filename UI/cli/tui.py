#!/usr/bin/env python3
"""CLI entry point for the MonaiUI Terminal UI."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Go-based TUI."""
    tui_binary = Path(__file__).parent.parent / "tui" / "automonai-tui"

    if not tui_binary.exists():
        print(f"Error: TUI binary not found at {tui_binary}", file=sys.stderr)
        print("Run 'cd UI/tui && go build -o automonai-tui .' to build it.", file=sys.stderr)
        sys.exit(1)

    print("Starting MonaiUI Terminal UI...\n")

    try:
        subprocess.run(
            [str(tui_binary)],
            check=False,
            cwd=str(Path(__file__).parent.parent.parent),
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
