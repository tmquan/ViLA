"""Registry-level smoke tests for the Family-B (HTML crawler) datasites.

`pbgdpl` and `phapdien` do NOT run under the NeMo Curator multi-stage
contract; they orchestrate via ``packages.common.runner.run_crawler_site``
and expose three module-level attributes per site:

* ``PIPELINES``           -- ``dict[str, Callable[[cfg], Path]]``
* ``ALL_PIPELINES_ORDER`` -- ``list[str]``
* ``run_pipeline``        -- ``Callable[[cfg, name], Path]`` (dispatcher
  that rejects unknown names; ``"all"`` is reserved for the runner).

The Family-A counterpart lives in ``test_pipeline_build.py``. The
contract these tests pin is documented in
``docs/00-overview/repo-layout.md`` under "Package boundaries".
"""

from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------- helpers


def _scraper_module(site: str) -> Any:
    """Return ``packages.datasites.<site>.scraper`` (lazy import)."""
    import importlib

    return importlib.import_module(f"packages.datasites.{site}.scraper")


CRAWLER_SITES: list[tuple[str, tuple[str, ...]]] = [
    ("pbgdpl", ("harvest", "detail")),
    ("phapdien", ("tree", "detail")),
    ("thuvienphapluat_tnpl", ("harvest", "detail", "translate")),
]


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("site,expected", CRAWLER_SITES)
def test_pipelines_registry_keys_match_documented_order(
    site: str, expected: tuple[str, ...]
) -> None:
    mod = _scraper_module(site)
    assert tuple(mod.PIPELINES.keys()) == expected, (
        f"{site}.scraper.PIPELINES has wrong keys; "
        f"docs/00-overview/repo-layout.md must be updated if this is intentional."
    )
    assert tuple(mod.ALL_PIPELINES_ORDER) == expected


@pytest.mark.parametrize("site,_expected", CRAWLER_SITES)
def test_every_pipeline_value_is_callable(site: str, _expected: tuple[str, ...]) -> None:
    mod = _scraper_module(site)
    for name, fn in mod.PIPELINES.items():
        assert callable(fn), f"{site}.scraper.PIPELINES[{name!r}] is not callable"


@pytest.mark.parametrize("site,_expected", CRAWLER_SITES)
def test_run_pipeline_rejects_unknown_name(site: str, _expected: tuple[str, ...]) -> None:
    mod = _scraper_module(site)
    with pytest.raises(ValueError):
        mod.run_pipeline(cfg=object(), name="__never_a_real_pipeline__")


@pytest.mark.parametrize("site,_expected", CRAWLER_SITES)
def test_run_pipeline_signature(site: str, _expected: tuple[str, ...]) -> None:
    """``run_pipeline(cfg, name)`` must be the dispatch surface."""
    import inspect

    mod = _scraper_module(site)
    sig = inspect.signature(mod.run_pipeline)
    params = list(sig.parameters)
    assert params[:2] == ["cfg", "name"], (
        f"{site}.scraper.run_pipeline must accept (cfg, name); got {params}"
    )


@pytest.mark.parametrize("site,_expected", CRAWLER_SITES)
def test_all_keyword_is_not_a_pipeline_name(site: str, _expected: tuple[str, ...]) -> None:
    """``"all"`` is reserved by ``run_crawler_site``; never a site pipeline name."""
    mod = _scraper_module(site)
    assert "all" not in mod.PIPELINES
    assert "all" not in mod.ALL_PIPELINES_ORDER
