"""Unit tests for :mod:`packages.common.runner`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from packages.common.runner import run_crawler_site, run_curator_site


class _StubPipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.run_called_with: Any = None

    def describe(self) -> str:
        return f"<pipeline {self.name}>"

    def run(self, executor: Any) -> list[Any]:
        self.run_called_with = executor
        return ["task-1", "task-2"]


def _site_config(tmp_path: Path) -> Path:
    """Write a minimal site config OmegaConf can parse + return its path."""
    cfg_path = tmp_path / "default.yaml"
    cfg_path.write_text(
        "host: example.test\n"
        f"output_dir: {tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    return cfg_path


def test_run_curator_site_invokes_build_pipeline_per_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _site_config(tmp_path)
    built: list[str] = []
    runs: list[str] = []

    def fake_build_pipeline(cfg: Any, name: str) -> _StubPipeline:
        built.append(name)
        return _StubPipeline(name)

    # Stub the heavy pipeline backend so the test doesn't need Ray.
    import packages.pipeline as ppl
    monkeypatch.setattr(ppl, "init_ray", lambda cfg: None)
    monkeypatch.setattr(ppl, "shutdown_ray", lambda: None)
    monkeypatch.setattr(ppl, "build_executor", lambda cfg: "executor-stub")

    rc = run_curator_site(
        site="example",
        pipelines={"a": "stub_a", "b": "stub_b"},
        all_order=["a", "b"],
        build_pipeline=fake_build_pipeline,
        description="test",
        argv=["--config", str(cfg_path), "--pipeline", "all"],
    )
    assert rc == 0
    assert built == ["a", "b"]


def test_run_curator_site_runs_single_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _site_config(tmp_path)
    built: list[str] = []

    def fake_build_pipeline(cfg: Any, name: str) -> _StubPipeline:
        built.append(name)
        return _StubPipeline(name)

    import packages.pipeline as ppl
    monkeypatch.setattr(ppl, "init_ray", lambda cfg: None)
    monkeypatch.setattr(ppl, "shutdown_ray", lambda: None)
    monkeypatch.setattr(ppl, "build_executor", lambda cfg: "executor-stub")

    rc = run_curator_site(
        site="example",
        pipelines={"a": "stub_a", "b": "stub_b"},
        all_order=["a", "b"],
        build_pipeline=fake_build_pipeline,
        argv=["--config", str(cfg_path), "--pipeline", "b"],
    )
    assert rc == 0
    assert built == ["b"]


def test_run_curator_site_returns_one_on_pipeline_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _site_config(tmp_path)

    def fake_build_pipeline(cfg: Any, name: str) -> _StubPipeline:
        raise RuntimeError("boom")

    import packages.pipeline as ppl
    monkeypatch.setattr(ppl, "init_ray", lambda cfg: None)
    monkeypatch.setattr(ppl, "shutdown_ray", lambda: None)
    monkeypatch.setattr(ppl, "build_executor", lambda cfg: "executor-stub")

    rc = run_curator_site(
        site="example",
        pipelines={"a": "stub_a"},
        all_order=["a"],
        build_pipeline=fake_build_pipeline,
        argv=["--config", str(cfg_path)],
    )
    assert rc == 1


def test_run_crawler_site_calls_run_pipeline_per_name(tmp_path: Path) -> None:
    cfg_path = _site_config(tmp_path)
    calls: list[str] = []

    def fake_run_pipeline(cfg: Any, name: str) -> Path:
        calls.append(name)
        return tmp_path / f"{name}.jsonl"

    rc = run_crawler_site(
        site="example",
        pipelines={"harvest": "h", "detail": "d"},
        all_order=["harvest", "detail"],
        run_pipeline=fake_run_pipeline,
        argv=["--config", str(cfg_path), "--pipeline", "all"],
    )
    assert rc == 0
    assert calls == ["harvest", "detail"]


def test_run_crawler_site_returns_one_on_failure(tmp_path: Path) -> None:
    cfg_path = _site_config(tmp_path)

    def fake_run_pipeline(cfg: Any, name: str) -> Path:
        raise RuntimeError("crawler error")

    rc = run_crawler_site(
        site="example",
        pipelines={"harvest": "h"},
        all_order=["harvest"],
        run_pipeline=fake_run_pipeline,
        argv=["--config", str(cfg_path)],
    )
    assert rc == 1


def test_per_site_main_modules_load_without_error() -> None:
    """Each datasite's runnable entry imports + exposes `main`.

    anle migrated its CLI to the in-process ``pipeline`` runner (the old
    top-level ``__main__`` moved to ``anle/_legacy/``); the Family-B
    harvesters and congbobanan keep a ``__main__`` shell.
    """
    from packages.datasites.anle import pipeline as anle_main
    from packages.datasites.congbobanan import __main__ as cong_main
    from packages.datasites.pbgdpl import __main__ as pbgdpl_main
    from packages.datasites.phapdien import __main__ as phap_main

    for mod in (anle_main, cong_main, pbgdpl_main, phap_main):
        assert callable(mod.main), f"{mod.__name__} must expose `main()`"
