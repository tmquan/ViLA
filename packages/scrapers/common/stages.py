"""StageBase: unified scaffolding for every pipeline stage.

Every ViLA scraper pipeline stage (scrape / parse / extract / embed /
reduce / visualize) needs the same seven things at startup:

    1. Hold the OmegaConf cfg.
    2. Hold the SiteLayout describing data/{host}/... paths.
    3. Honor `--force` and `--no-resume` flags.
    4. Honor `--limit` (when applicable).
    5. Ensure the stage's output directories exist.
    6. Open a per-stage JSONL log file.
    7. Optionally maintain a per-item `progress.<stage>.json` checkpoint.

This module captures that once. A concrete stage declares:

    class AnleParser(StageBase):
        stage = "parse"                                # -> progress file name
        required_dirs = ("md_dir", "json_dir",         # Layout attribute names
                         "metadata_dir", "logs_dir")
        uses_progress = True

        def run(self) -> dict[str, int]:
            ...

That is the entire boilerplate. Subclasses only add stage-specific
fields (e.g. `client` for the parser, `registry` for the embedder)
and the `run()` implementation.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.logging import SiteLogger
from packages.scrapers.common.progress import ProgressState

logger = logging.getLogger(__name__)


class StageBase(abc.ABC):
    """Common infrastructure for every pipeline stage.

    Class attributes (declared by subclasses):
        stage (str):              Stage name — one of
                                  'scrape','parse','extract','embed',
                                  'reduce','visualize'. Used for the
                                  progress file name and log tag.
        required_dirs (tuple):    Names of SiteLayout attributes
                                  (e.g. 'md_dir', 'logs_dir') that MUST
                                  exist on disk before `run()`.
        uses_progress (bool):     When True, a ProgressState checkpoint
                                  is loaded at init and reset by
                                  `resume=False`. Stages whose outputs
                                  are idempotent per file (reducer,
                                  visualizer) can set this False.

    Instance attributes:
        cfg                       The OmegaConf DictConfig for the pipeline.
        layout                    The SiteLayout.
        force (bool)              Redo all work regardless of checkpoint.
        limit (int | None)        Process at most N items.
        log (SiteLogger)          Append-only JSONL event logger.
        progress (ProgressState | None)
                                  Set iff `uses_progress` is True.
    """

    stage: str = ""
    required_dirs: tuple[str, ...] = ("site_root", "logs_dir")
    uses_progress: bool = True

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        *,
        force: bool = False,
        resume: bool = True,
        limit: int | None = None,
    ) -> None:
        if not self.stage:
            raise TypeError(
                f"{type(self).__name__} must set the `stage` class attribute."
            )
        self.cfg = cfg
        self.layout = layout
        self.force = force
        self.limit = limit

        dirs = [getattr(self.layout, name) for name in self.required_dirs]
        self.layout.ensure_dirs(*dirs)

        self.log = SiteLogger(log_dir=self.layout.logs_dir, stage=self.stage)

        if self.uses_progress:
            self.progress: ProgressState | None = ProgressState(
                path=self.layout.progress_path(self.stage),
                stage=self.stage,
            )
            if not resume:
                self.progress.reset()
        else:
            self.progress = None

    @abc.abstractmethod
    def run(self) -> dict[str, int]:
        """Execute the stage. Return summary counts for logging."""
