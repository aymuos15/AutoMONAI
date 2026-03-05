#!/usr/bin/env python3
"""CLI entry point for the MonaiUI web GUI."""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main():
    ui_dir = Path(__file__).parent.parent
    server_script = ui_dir / "server.py"

    print("Starting MonaiUI Web GUI...\n")

    try:
        # Give server a moment to start, then open browser
        proc = subprocess.Popen([sys.executable, str(server_script)], cwd=str(ui_dir.parent))

        # Wait for server to start
        time.sleep(2)

        # Try to open browser
        try:
            webbrowser.open("http://localhost:8888")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print("   Please open http://localhost:8888 manually\n")

        # Wait for process
        proc.wait()

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        proc.terminate()
        proc.wait()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
