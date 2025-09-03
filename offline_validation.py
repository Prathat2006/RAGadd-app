import shutil
import subprocess
import sys
import json
import socket
from urllib.request import urlopen, Request
from urllib.error import URLError
from contextlib import closing

def _has_executable(name: str) -> bool:
    return shutil.which(name) is not None

def _runs_successfully(cmd: list[str]) -> bool:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return p.returncode == 0
    except Exception:
        return False

def _http_ok(url: str, expect_json=False) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "install-check/1.0"})
        with closing(urlopen(req, timeout=3)) as r:
            if expect_json:
                try:
                    json.load(r)
                    return True
                except Exception:
                    return False
            else:
                # Any 200 OK is fine; some endpoints return a small text banner
                return 200 <= r.getcode() < 300
    except URLError:
        return False
    except Exception:
        return False

def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except Exception:
        return False

def is_ollama_installed() -> bool:
    """
    True if Ollama CLI is callable OR the local Ollama server responds.
    """
    # 1) CLI presence
    if _has_executable("ollama") and _runs_successfully(["ollama", "--version"]):
        return True

    # 2) Server banner / health (works whether started by desktop app or 'ollama serve')
    #    Root returns 'Ollama is running' when active; API commonly at /api/*
    for host in ("127.0.0.1", "localhost"):
        if _port_open(host, 11434):
            if _http_ok(f"http://{host}:11434/"):           # health/banner
                return True
            if _http_ok(f"http://{host}:11434/api/tags", expect_json=True):  # list models
                return True
    return False

def is_lmstudio_installed(base_url: str | None = None) -> bool:
    """
    True if LM Studio CLI ('lms') is callable OR its local server responds.
    base_url: override like 'http://127.0.0.1:1234' if you changed the default port.
    """
    # 1) CLI presence (LM Studio ships a CLI called 'lms')
    #    Either 'lms --version' or 'lms status' should work once LM Studio has been run at least once.
    if _has_executable("lms") and (_runs_successfully(["lms", "--version"]) or _runs_successfully(["lms", "status"])):
        return True

    # 2) Server check (OpenAI-compatible API, default port 1234)
    candidates = []
    if base_url:
        candidates.append(base_url.rstrip("/"))
    else:
        candidates.extend([f"http://{h}:1234" for h in ("127.0.0.1", "localhost")])

    for root in candidates:
        # quickest reliable endpoint is GET /v1/models which returns JSON
        if _http_ok(f"{root}/v1/models", expect_json=True):
            return True

    return False

# if __name__ == "__main__":
#     print(is_ollama_installed())
#     print(is_lmstudio_installed())
