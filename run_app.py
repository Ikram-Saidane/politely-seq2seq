"""Start the Politely web interface and open it in the default browser.

    python run_app.py            (port 8765)
    python run_app.py --port 8080 --no-browser
"""
import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    if not any((ROOT / "checkpoints").glob("*.pt")):
        print("No trained model in checkpoints/. Train one first, e.g.:\n"
              "  python -m politely.train --config configs/B_attention.json   (with PYTHONPATH=src)")
    url = f"http://127.0.0.1:{a.port}"
    print(f"Politely is starting on {url}  (press Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.server:app", host="127.0.0.1", port=a.port, app_dir=str(ROOT), log_level="warning")


if __name__ == "__main__":
    main()
