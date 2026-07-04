"""dichvucong — NEW national portal (dichvucong.gov.vn /api/v1).

Playwright-driven crawler for the 2025+ SPA whose public ``/api/v1``
endpoints return the **full structured procedure detail** (executionSteps,
profileComponents, fees, legalBasis, results, agencies, …) for every
agency nationwide. No VNeID/login; the F5/TSPD WAF is solved by issuing
the JSON calls from a real browser context. See
``packages/datasites/dichvucong/README.md``.

    python -m packages.datasites.dichvucong --pipeline all
"""

from __future__ import annotations

from packages.datasites.dichvucong.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)

__all__ = ["ALL_PIPELINES_ORDER", "PIPELINES", "run_pipeline"]
