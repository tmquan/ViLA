"""Shared pipeline infrastructure used by every datasite + stage package.

Public surface:
    SiteLayout          - per-host filesystem layout (data/<host>/...).
    PoliteSession       - requests.Session wrapper with rate limit,
                          polite headers, SOCKS5/HTTP proxy support,
                          exponential-backoff retry. Used by site-level
                          URLGenerator + DocumentDownloader.
    SiteLogger          - JSONL operational logger (used by stages that
                          want a structured sidecar log next to outputs).
    build_arg_parser    - shared CLI flags: --executor, --ray-address,
                          --config, --override, --limit, --output.
    find_site_config    - resolve --config-name NAME against
                          packages/datasites/<name>/configs/<name>.yaml.
    PipelineCfg         - top-level OmegaConf-structured schema.
    ExecutorCfg / RayCfg - executor + Ray-client config.
"""

from packages.common.base import SiteLayout
from packages.common.cli import (
    EXECUTOR_CHOICES,
    apply_log_level,
    build_arg_parser,
    load_and_override,
)
from packages.common.config import (
    apply_overrides,
    find_site_config,
    load_config,
    resolve_config_path,
    resolve_stage_config,
    structured_config,
    to_container,
)
from packages.common.http import PoliteSession
from packages.common.logging import SiteLogger
from packages.common.schemas import ExecutorCfg, PipelineCfg, RayCfg

__all__ = [
    "EXECUTOR_CHOICES",
    "ExecutorCfg",
    "PipelineCfg",
    "PoliteSession",
    "RayCfg",
    "SiteLayout",
    "SiteLogger",
    "apply_log_level",
    "apply_overrides",
    "build_arg_parser",
    "find_site_config",
    "load_and_override",
    "load_config",
    "resolve_config_path",
    "resolve_stage_config",
    "structured_config",
    "to_container",
]
