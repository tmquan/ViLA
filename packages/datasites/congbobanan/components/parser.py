"""congbobanan Parser component — NeMo Curator ``PdfParseStage``, in-process.

Wraps :class:`packages.parser.stage.PdfParseStage` (runtime=local → ``PypdfParser``
with automatic Adobe-Vietnamese ToUnicode CMap healing) — the *same* Curator
``ProcessingStage`` anle's ``build_parse_pipeline`` uses. We drive it in-process
over ``DocumentBatch`` tasks (like anle's embed/reduce) so extraction runs
sharded + resumable without the xenna executor.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.parser.stage import PdfParseStage


def build_parser_cfg(runtime: str = "local") -> Any:
    return OmegaConf.create(
        {"parser": {"runtime": runtime, "model_id": "local/pypdf", "preserve_tables": True}}
    )


class CongbobananParser:
    """The Curator :class:`PdfParseStage`, driven in-process (local pypdf)."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg if cfg is not None else build_parser_cfg()
        self.stage = PdfParseStage(cfg=self.cfg)
        self.stage.setup(None)

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` has ``doc_name`` + ``pdf_bytes``; returns rows with non-empty
        ``markdown`` (the stage drops empty/unparseable rows internally)."""
        batch = DocumentBatch(task_id="cbb_parse", dataset_name="congbobanan",
                              data=df.reset_index(drop=True))
        return self.stage.process(batch).to_pandas()


__all__ = ["CongbobananParser", "build_parser_cfg"]
