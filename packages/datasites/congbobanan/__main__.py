"""CLI entry for the congbobanan download+extract runner.

Delegates to the single-IP paced runner in
:mod:`packages.datasites.congbobanan.pipeline`::

    python -m packages.datasites.congbobanan --start 1 --end 100
    python -m packages.datasites.congbobanan --url-list urls.txt --proxy http://vn-egress:3128

congbobanan.toaan.gov.vn rejects non-VN source IPs; pass a VN-egress
``--proxy`` or run on a VN VPS.
"""

from __future__ import annotations

import sys

from packages.datasites.congbobanan.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
