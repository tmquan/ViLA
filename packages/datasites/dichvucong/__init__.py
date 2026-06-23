"""dichvucong datasite — Cổng Dịch vụ công Quốc gia (national TTHC).

Source: https://dichvucong.gov.vn/ (run by Văn phòng Chính phủ). The
administrative-procedure (thủ tục hành chính) corpus is served by a
single JSON gateway, ``POST /jsp/rest.jsp``, which **already aggregates
every ministry and province** — including Bộ Công An
(``dichvucong.bocongan.gov.vn``) — so one datasite covers them all.

See ``wiki/DICHVUCONG.md`` for the API contract and the incremental
freshness mechanism.

Run via::

    python -m packages.datasites.dichvucong --pipeline all
    python -m packages.datasites.dichvucong.reconcile
"""

from __future__ import annotations

from packages.datasites.dichvucong.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "build_pipeline",
]
