"""SiteScraperBase: base class for per-site scrapers (stage 1).

Mirrors the public API used by the datascraper reference repo so that
operators running different sites see identical semantics:

    * load_progress / save_progress - handled by ProgressState via StageBase
    * is_item_complete(item_id)     - inspect filesystem
    * process_item(item_ref)        - fetch + store one item
    * run(items_iter)               - orchestrate across a ThreadPool

Subclasses implement the three abstract methods `iter_items`,
`item_id`, `is_item_complete`, and `process_item`. Everything else
(rate limit, progress persistence, logging, resume) is provided by the
base via StageBase.
"""

from __future__ import annotations

import abc
import concurrent.futures as cf
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from packages.scrapers.common.http import PoliteSession

logger = logging.getLogger(__name__)


@dataclass
class SiteLayout:
    """Filesystem layout for one site under data/{host}/..."""

    output_root: Path
    host: str

    @property
    def site_root(self) -> Path:
        return self.output_root / self.host

    @property
    def pdf_dir(self) -> Path:
        return self.site_root / "pdf"

    @property
    def md_dir(self) -> Path:
        return self.site_root / "md"

    @property
    def json_dir(self) -> Path:
        return self.site_root / "json"

    @property
    def jsonl_dir(self) -> Path:
        return self.site_root / "jsonl"

    @property
    def parquet_dir(self) -> Path:
        return self.site_root / "parquet"

    @property
    def metadata_dir(self) -> Path:
        return self.site_root / "metadata"

    @property
    def viz_dir(self) -> Path:
        return self.site_root / "viz"

    @property
    def logs_dir(self) -> Path:
        return self.site_root / "logs"

    def progress_path(self, stage: str) -> Path:
        """One progress file per stage (avoids resume conflicts)."""
        return self.site_root / f"progress.{stage}.json"

    def ensure_dirs(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


class SiteScraperBase:
    """Per-site scraper base class (stage 1).

    Lifecycle:
        1. __init__       -> load config, init session, progress, log
        2. iter_items()   -> yield item_refs (typed by subclass)
        3. run()          -> thread-pool across items; calls
                             is_item_complete(), then process_item()

    Resume semantics:
        - If item_id in progress.completed AND is_item_complete(id)
          -> skip.
        - If progress says complete but filesystem disagrees
          (partial download, interrupted write) -> re-run.
        - --no-resume clears the progress set.
        - --force bypasses is_item_complete entirely.

    This class is intentionally NOT a StageBase subclass because it
    exists in the same module as SiteLayout; importing StageBase here
    would create a circular dependency. Instead it reproduces the same
    scaffolding directly. The five disk stages under packages/scrapers/
    anle/ *do* inherit from StageBase.
    """

    stage: str = "scrape"
    required_dirs: tuple[str, ...] = (
        "site_root",
        "pdf_dir",
        "metadata_dir",
        "logs_dir",
    )

    def __init__(
        self,
        layout: SiteLayout,
        session: PoliteSession,
        *,
        num_workers: int = 4,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
    ) -> None:
        # Lazy import to avoid circular dependency with stages.py.
        from packages.scrapers.common.logging import SiteLogger
        from packages.scrapers.common.progress import ProgressState

        self.layout = layout
        self.session = session
        self.num_workers = num_workers
        self.limit = limit
        self.force = force

        dirs = [getattr(self.layout, name) for name in self.required_dirs]
        self.layout.ensure_dirs(*dirs)

        self.progress = ProgressState(
            path=self.layout.progress_path(self.stage),
            stage=self.stage,
        )
        if not resume:
            self.progress.reset()
        self.log = SiteLogger(log_dir=self.layout.logs_dir, stage=self.stage)

    # ------------------------------------------------------------------ ABC

    @abc.abstractmethod
    def iter_items(self) -> Iterator[Any]:
        """Yield item references (type chosen by subclass)."""

    @abc.abstractmethod
    def item_id(self, item: Any) -> str:
        """Return the stable string ID for an item reference."""

    @abc.abstractmethod
    def is_item_complete(self, item_id: str) -> bool:
        """Return True if outputs for this item already exist on disk."""

    @abc.abstractmethod
    def process_item(self, item: Any) -> dict[str, Any]:
        """Fetch + persist one item. Return a record dict for data.csv."""

    # -------------------------------------------------------------- lifecycle

    def run(self) -> dict[str, int]:
        """Run the scraper across all items. Return summary counts."""
        counts = {"seen": 0, "skipped": 0, "processed": 0, "errored": 0}
        futures: list[cf.Future[dict[str, Any] | None]] = []
        with cf.ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for item in self._iter_with_limit():
                counts["seen"] += 1
                item_id = self.item_id(item)
                if (
                    not self.force
                    and self.progress.is_complete(item_id)
                    and self.is_item_complete(item_id)
                ):
                    counts["skipped"] += 1
                    continue
                futures.append(ex.submit(self._process_one, item))
            for fut in cf.as_completed(futures):
                try:
                    record = fut.result()
                except Exception as exc:
                    counts["errored"] += 1
                    self.log.error(error=str(exc))
                    logger.exception("process_item failed")
                else:
                    if record is not None:
                        counts["processed"] += 1
        self.log.info(event="run_done", **counts)
        return counts

    def _iter_with_limit(self) -> Iterable[Any]:
        n = 0
        for item in self.iter_items():
            if self.limit is not None and n >= self.limit:
                return
            yield item
            n += 1

    def _process_one(self, item: Any) -> dict[str, Any] | None:
        item_id = self.item_id(item)
        try:
            record = self.process_item(item)
        except Exception as exc:
            self.log.error(item_id=item_id, error=str(exc))
            raise
        else:
            self.progress.mark_complete(item_id)
            self.log.info(item_id=item_id, event="complete")
            return record
