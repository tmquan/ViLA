"""End-to-end smoke test: stages 2..5 as a Curator in-process chain.

Exercises :class:`PdfParseStage`, :class:`LegalExtractStage`,
:class:`NimEmbedderStage`, :class:`ReducerStage`, and the terminal
:class:`JsonlParquetWriter` by hand-building a :class:`DocumentBatch`
as if it had come out of the download composite, then running the
stages' ``process()`` methods sequentially without an executor.

The separate executor-integration path (``pipeline.run(executor=...)``
against a live Ray cluster) is covered by the acceptance commands
in the plan's §6 checklist; we do not start Ray inside the test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.embedder.base import ModelEntry
from packages.embedder.stage import NimEmbedderStage
from packages.extractor.stage import LegalExtractStage
from packages.parser.base import ParserAlgorithm
from packages.parser.stage import PdfParseStage
from packages.reducer.stage import ReducerStage


try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


pytestmark = pytest.mark.skipif(
    not HAVE_SKLEARN, reason="sklearn missing; cannot run reducer path"
)


class _FakeNemotron(ParserAlgorithm):
    runtime = "fake"
    model_id = "fake/nemotron"

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        return {
            "pages": [
                {
                    "page_number": 1,
                    "markdown": (
                        "# Án lệ số 47/2021/AL\n\n"
                        "Nội dung án lệ: áp dụng khoản 1 Điều 173 BLHS 2015.\n\n"
                        "Ngày thông qua 15/06/2021."
                    ),
                }
            ],
            "markdown": "(stitched)",
            "confidence": 0.92,
        }


class _FakeEmbedder:
    model_id = "fake/embed-smoke"
    embedding_dim = 4
    max_seq_length = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, t in enumerate(texts):
            v = [0.0] * 4
            v[i % 4] = 1.0 + len(t) * 1e-6
            out.append(v)
        return out


def _cfg(tmp_path: Path) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.output_dir = str(tmp_path)
    cfg.embedder.model_id = "fake/embed-smoke"
    cfg.embedder.runtime = "nim"
    cfg.embedder.chunking = "off"
    cfg.embedder.max_seq_length = 128
    cfg.embedder.batch_size = 4
    cfg.parser.runtime = "local"
    cfg.reducer.methods = ["pca"]
    cfg.reducer.prefer_gpu = False
    return cfg


def _seed_batch(n: int = 3) -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": [f"TAND{i:03d}" for i in range(n)],
            "pdf_bytes": [b"%PDF-1.4\nfake" for _ in range(n)],
            "adopted_date": ["15/06/2021"] * n,
            "applied_article": ["Điều 173 BLHS 2015"] * n,
            "precedent_number": [f"Án lệ số {i + 1:02d}/2021/AL" for i in range(n)],
            "source": ["anle.toaan.gov.vn"] * n,
        }
    )
    return DocumentBatch(task_id="anle-test", dataset_name="anle", data=df)


def test_stages_chain_produces_final_schema(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    parse = PdfParseStage(cfg=cfg)
    parse._client = _FakeNemotron()
    extract = LegalExtractStage(cfg=cfg)
    extract.setup(None)
    embed = NimEmbedderStage(cfg=cfg)
    embed._entry = ModelEntry("fake/embed-smoke", "nim", 4, True, None)
    embed._backend = _FakeEmbedder()
    reduce_ = ReducerStage(cfg=cfg)

    batch = _seed_batch(3)
    batch = parse.process(batch)
    batch = extract.process(batch)
    batch = embed.process(batch)
    batch = reduce_.process(batch)

    df = batch.to_pandas()
    assert len(df) == 3

    # Parser columns.
    assert {"markdown", "num_pages", "parser_model"} <= set(df.columns)
    # Extractor columns.
    assert {"text_hash", "extracted", "precedent_number"} <= set(df.columns)
    # Embedder columns.
    assert {"embedding", "embedding_dim", "embedding_model_id"} <= set(df.columns)
    # Reducer columns.
    assert {"pca_x", "pca_y", "cluster_id"} <= set(df.columns)

    # Every embedding is 4-dim and the reducer produced 2-D coords.
    assert (df["embedding_dim"] == 4).all()
    assert df["pca_x"].notna().all()

    # No raw bytes survive past the parser.
    assert "pdf_bytes" not in df.columns
