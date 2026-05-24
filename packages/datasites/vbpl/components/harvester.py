"""Walk vbpl.vn's public sitemap chain and write ``jsonl/sitemap.jsonl``.

vbpl publishes a sitemap index at ``https://vbpl.vn/sitemap.xml`` that
points at:

* ``sitemap-static.xml``                 (skipped: index/landing pages)
* ``sitemap-trung-uong-{1..11}.xml``     (~5 000 detail URLs each, central docs)
* ``sitemap-dia-phuong-{1..21}.xml``     (~5 000 detail URLs each, provincial docs)

Each ``<url>`` row is one ``/van-ban/chi-tiet/<slug>--<numeric-id>``
detail page. We don't hit any backend gateway here; the entire harvest
runs against vbpl.vn's static sitemap files (allowed by ``robots.txt``)
with a single :class:`packages.common.PoliteSession`.

The shards are cached as raw XML under ``html/sitemaps/<filename>.xml``
so a re-run only repays the cost of the index page (it changes any
time vbpl re-publishes a shard, and lastmod tells us nothing about
which shard changed). Set ``cfg.scraper.cache_listings=false`` to
force a clean re-fetch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.vbpl._shared import (
    SCOPES,
    SITEMAP_JSONL_FIELDS,
    sitemaps_dir,
)
from packages.datasites.vbpl.components.parser import (
    SitemapEntry,
    parse_sitemap_index,
    parse_sitemap_urlset,
    scope_from_shard_url,
)

logger = logging.getLogger(__name__)


DEFAULT_SITEMAP_URL = "https://vbpl.vn/sitemap.xml"


class VbplSitemapHarvester:
    """Top-level orchestrator for the sitemap walk.

    Stores only pickle-safe state on the driver (cfg + URLs). The
    :class:`PoliteSession` is built lazily inside :meth:`run` so the
    object is safe to instantiate before the rate-limited session
    burns its first token.
    """

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._sitemap_url: str = (
            str(cfg.scraper.get("sitemap_url", "")) or DEFAULT_SITEMAP_URL
        )
        self._cache_listings: bool = bool(
            cfg.scraper.get("cache_listings", True),
        )
        scopes_cfg = list(cfg.scraper.get("scopes", []) or [])
        if scopes_cfg:
            for s in scopes_cfg:
                if s not in SCOPES:
                    raise ValueError(
                        f"unknown scope {s!r} in cfg.scraper.scopes; "
                        f"valid: {SCOPES}",
                    )
            self._scopes_filter: set[str] | None = set(scopes_cfg)
        else:
            self._scopes_filter = None  # None => all known scopes
        self._session: PoliteSession | None = None

    # ------------------------------------------------------ entrypoint

    def run(self) -> tuple[Path, int]:
        """Walk every shard and write ``sitemap.jsonl``.

        Returns ``(sitemap_jsonl_path, total_rows_written)``.
        """
        if self._session is None:
            self._session = session_from_scraper_cfg(self.cfg)

        index_xml = self._fetch_index()
        shards = parse_sitemap_index(index_xml)
        if not shards:
            raise RuntimeError(
                f"sitemap index empty or unparseable at {self._sitemap_url}",
            )

        # Filter to in-scope shards. ``sitemap-static.xml`` and any
        # other unknown layout never carry detail pages.
        scoped: list[tuple[str, str]] = []
        for url in shards:
            scope = scope_from_shard_url(url)
            if scope is None:
                logger.info("skipping non-detail shard: %s", url)
                continue
            if self._scopes_filter is not None and scope not in self._scopes_filter:
                logger.info("skipping out-of-scope shard: %s (scope=%s)", url, scope)
                continue
            scoped.append((url, scope))
        logger.info(
            "harvest: %d in-scope shards (filter=%s)",
            len(scoped),
            sorted(self._scopes_filter) if self._scopes_filter else "<all>",
        )

        out_path = self.layout.jsonl_dir / "sitemap.jsonl"
        harvested_at = _utc_now_iso()
        total = 0
        per_scope: dict[str, int] = {s: 0 for s in SCOPES}
        with out_path.open("w", encoding="utf-8") as f:
            for shard_url, scope in scoped:
                xml_text = self._fetch_shard(shard_url)
                entries = parse_sitemap_urlset(xml_text, scope=scope)
                if not entries:
                    logger.warning("shard empty: %s", shard_url)
                    continue
                for entry in entries:
                    f.write(_entry_to_jsonl(entry, harvested_at))
                    f.write("\n")
                total += len(entries)
                per_scope[scope] = per_scope.get(scope, 0) + len(entries)
                logger.info(
                    "shard ok: %s (+%d rows; total=%d)",
                    Path(shard_url).name, len(entries), total,
                )

        logger.info(
            "sitemap.jsonl written: rows=%d %s -> %s",
            total,
            ", ".join(f"{k}={v}" for k, v in per_scope.items() if v),
            out_path,
        )
        return out_path, total

    # ------------------------------------------------------ HTTP

    def _fetch_index(self) -> str:
        """GET the sitemap index. Cached under ``html/sitemaps/index.xml``."""
        cache = sitemaps_dir(self.layout) / "index.xml"
        if (
            self._cache_listings
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")
        assert self._session is not None
        resp = self._session.get(self._sitemap_url)
        if resp.status_code != 200:
            raise RuntimeError(
                f"sitemap index fetch failed: status={resp.status_code} "
                f"url={self._sitemap_url}",
            )
        text = resp.text or ""
        if self._cache_listings:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text

    def _fetch_shard(self, shard_url: str) -> str:
        """GET one shard. Cached under ``html/sitemaps/<basename>``."""
        cache = sitemaps_dir(self.layout) / Path(shard_url).name
        if (
            self._cache_listings
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")
        assert self._session is not None
        resp = self._session.get(shard_url)
        if resp.status_code != 200:
            logger.warning(
                "shard fetch failed: status=%d url=%s",
                resp.status_code, shard_url,
            )
            return ""
        text = resp.text or ""
        if self._cache_listings:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text


# ---- helpers -------------------------------------------------------------


def _entry_to_jsonl(entry: SitemapEntry, harvested_at: str) -> str:
    row: dict[str, Any] = {
        "item_id": entry.item_id,
        "scope": entry.scope,
        "slug": entry.slug,
        "url": entry.url,
        "lastmod": entry.lastmod,
        "changefreq": entry.changefreq,
        "priority": entry.priority,
        "harvested_at": harvested_at,
    }
    return json.dumps(
        {k: row.get(k) for k in SITEMAP_JSONL_FIELDS},
        ensure_ascii=False,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_SITEMAP_URL",
    "VbplSitemapHarvester",
]
