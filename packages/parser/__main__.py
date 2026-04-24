"""Allow ``python -m packages.parser``."""

from __future__ import annotations

import sys

from packages.parser.stage import main

if __name__ == "__main__":
    sys.exit(main())
