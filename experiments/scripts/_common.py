"""Script preamble: put ``src/`` on the path and load repo-root ``.env``.

SAGE-Eval is a gated HuggingFace dataset (needs ``HF_TOKEN``) and the judge
needs ``GEMINI_API_KEY``. Importing this first means the scripts work whether
those are exported or sitting in a ``.env``.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# First existing file wins. Already-set vars are never overwritten, so an
# explicit export always takes precedence.
ENV_CANDIDATES = (ROOT / ".env", ROOT.parent / ".env")


def load_env() -> Path | None:
    for path in ENV_CANDIDATES:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return path
    return None


load_env()
