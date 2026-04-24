"""Unit tests for :class:`PdfParseStage` with a fake parser backend."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.parser.base import ParserAlgorithm
from packages.parser.stage import PdfParseStage


class FakeNemotronClient(ParserAlgorithm):
    """Returns canned per-page output without touching the network."""

    runtime = "fake"
    model_id = "fake/nemotron"

    def __init__(self, pages: int = 2, confidence: float = 0.9) -> None:
        self._pages = pages
        self._confidence = confidence

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        return {
            "pages": [
                {
                    "page_number": i + 1,
                    "blocks": [{"type": "Title", "text": f"Page {i + 1}"}],
                    "markdown": f"# Page {i + 1}\nĐiều 173 BLHS 2015.",
                }
                for i in range(self._pages)
            ],
            "markdown": "# Page 1\nĐiều 173 BLHS 2015.",
            "confidence": self._confidence,
        }


def _cfg() -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "local"  # avoid NIM API key probe
    cfg.parser.model_id = "fake/nemotron"
    return cfg


def _make_batch(n: int = 2) -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": [f"TAND{i}" for i in range(n)],
            "pdf_bytes": [b"%PDF-1.4\n..." for _ in range(n)],
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


def test_process_returns_new_batch_with_parser_columns() -> None:
    stage = PdfParseStage(cfg=_cfg())
    stage._client = FakeNemotronClient(pages=3)  # bypass setup()

    out = stage.process(_make_batch(2))
    df = out.to_pandas()

    assert list(df["doc_name"]) == ["TAND0", "TAND1"]
    assert all(df["num_pages"] == 3)
    assert all(m.startswith("# Page 1") for m in df["markdown"])
    assert (df["parser_model"] == "fake/nemotron").all()
    assert "pdf_bytes" not in df.columns, "raw bytes must be dropped downstream"


def test_process_advertises_inputs_outputs() -> None:
    stage = PdfParseStage(cfg=_cfg())
    in_attrs, in_cols = stage.inputs()
    out_attrs, out_cols = stage.outputs()
    assert in_attrs == ["data"]
    assert "pdf_bytes" in in_cols
    assert out_attrs == ["data"]
    assert {"markdown", "pages", "confidence", "num_pages"} <= set(out_cols)


def test_process_handles_str_bytes_from_parquet_roundtrip() -> None:
    stage = PdfParseStage(cfg=_cfg())
    stage._client = FakeNemotronClient(pages=1)

    # Parquet sometimes demotes bytes to latin-1 strings; the stage
    # must recover gracefully.
    df = pd.DataFrame({"doc_name": ["X"], "pdf_bytes": ["%PDF-1.4"]})
    batch = DocumentBatch(task_id="t", dataset_name="anle", data=df)

    out = stage.process(batch)
    assert out.to_pandas()["num_pages"].iloc[0] == 1
