"""Shared scraper infrastructure used by per-site packages.

Public surface:
    SiteScraperBase     - base class implementing the datascraper shape
                          (load_progress/save_progress/is_item_complete/
                          process_item/run).
    ProgressState       - persistent resume checkpoint (progress.json).
    build_arg_parser    - shared CLI flags: --num-workers, --limit,
                          --no-resume, --output, --config, --proxy,
                          --force, --stop-after.
    PoliteSession       - requests.Session wrapper with rate limit,
                          polite headers, SOCKS5/HTTP proxy support,
                          exponential-backoff retry.
    SiteLogger          - JSONL per-stage operational logger.
"""

from packages.scrapers.common.base import SiteLayout, SiteScraperBase
from packages.scrapers.common.cli import build_arg_parser, load_and_override
from packages.scrapers.common.config import (
    apply_overrides,
    load_config,
    resolve_config_path,
    structured_config,
    to_container,
)
from packages.scrapers.common.http import PoliteSession
from packages.scrapers.common.logging import SiteLogger
from packages.scrapers.common.progress import ProgressState
from packages.scrapers.common.stages import StageBase

__all__ = [
    "SiteLayout",
    "SiteScraperBase",
    "StageBase",
    "ProgressState",
    "PoliteSession",
    "SiteLogger",
    "build_arg_parser",
    "load_and_override",
    "load_config",
    "apply_overrides",
    "to_container",
    "structured_config",
    "resolve_config_path",
]
