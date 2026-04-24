"""Allow ``python -m packages.embedder``."""

from __future__ import annotations

import sys

from packages.embedder.stage import main

if __name__ == "__main__":
    sys.exit(main())
