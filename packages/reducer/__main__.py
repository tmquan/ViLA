"""Allow ``python -m packages.reducer``."""

from __future__ import annotations

import sys

from packages.reducer.stage import main

if __name__ == "__main__":
    sys.exit(main())
