"""OmegaConf-based config loader for scraper pipeline stages.

Design (PyTorch-Lightning / Hydra-ish flavor, without full Hydra):

    * YAML files under configs/, hierarchical via `_base:` inheritance
      (hfdata convention).
    * Typed structured schemas via OmegaConf.structured(@dataclass).
    * Hydra-style dotlist CLI overrides applied AFTER load:
        --override embedder.batch_size=16 num_workers=8
    * All consumers get a DictConfig with attribute access
      (cfg.embedder.model_id) and interpolation (${..foo}) resolved on
      demand.

Public surface:
    load_config(path)                 -> DictConfig with _base resolved
    apply_overrides(cfg, overrides)   -> DictConfig with dotlist merged
    to_container(cfg, resolve=True)   -> plain dict (for pydantic / JSON)
    structured_config(schema_cls)     -> DictConfig from a dataclass schema
    resolve_config_path(...)          -> pick a file from --config / --config-name

Example:
    cfg = load_config("packages/scrapers/anle/configs/anle.yaml")
    cfg = apply_overrides(cfg, ["embedder.batch_size=16"])
    print(cfg.embedder.model_id)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(path: Path | str) -> DictConfig:
    """Load a YAML config with `_base:` inheritance resolved via OmegaConf.merge.

    The child file overrides the base file. `_base` is popped from the
    returned config. Multiple levels of inheritance are supported (a base
    may itself declare `_base`).
    """
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise ValueError(f"Config {path} must be a YAML mapping at the top level.")
    base_name = raw.pop("_base", None)
    if base_name is not None:
        base_path = (path.parent / str(base_name)).resolve()
        base = load_config(base_path)
        return OmegaConf.merge(base, raw)  # type: ignore[return-value]
    return raw


def apply_overrides(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    """Apply Hydra-style dotlist overrides (e.g. 'embedder.batch_size=16').

    Overrides use the same syntax as Hydra's command-line overrides:
        key=value                    # set scalar
        section.nested.key=value     # nested dotted path
        key=[1,2,3]                  # list literal
        key=null                     # unset
    """
    override_list = list(overrides)
    if not override_list:
        return cfg
    override_cfg = OmegaConf.from_dotlist(override_list)
    return OmegaConf.merge(cfg, override_cfg)  # type: ignore[return-value]


def to_container(cfg: DictConfig | ListConfig, resolve: bool = True) -> Any:
    """Convert an OmegaConf cfg to a plain Python container (dict / list)."""
    return OmegaConf.to_container(cfg, resolve=resolve)


def structured_config(schema_cls: type) -> DictConfig:
    """Create a DictConfig from a dataclass schema for typed defaults.

    Use as the top of a merge chain:
        base = structured_config(PipelineCfg)
        file_cfg = load_config("configs/anle.yaml")
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base, file_cfg, cli_cfg)
    """
    return OmegaConf.structured(schema_cls)


def resolve_config_path(
    config: Path | None,
    config_name: str | None,
    configs_dir: Path,
    default_name: str = "default",
) -> Path:
    """Resolve a config path from --config or --config-name.

    Precedence: explicit --config > --config-name > default_name.
    """
    if config is not None:
        return Path(config).expanduser().resolve()
    name = config_name or default_name
    return (configs_dir / f"{name}.yaml").resolve()
