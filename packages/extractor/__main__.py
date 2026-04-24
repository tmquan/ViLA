"""Allow ``python -m packages.extractor``."""

from __future__ import annotations

import sys

from packages.extractor.stage import main

if __name__ == "__main__":
    sys.exit(main())
