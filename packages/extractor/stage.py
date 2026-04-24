"""Stage 3: legal extractor as a Curator :class:`ProcessingStage`.

Consumes a :class:`DocumentBatch` with a ``markdown`` column produced
by :class:`~packages.parser.stage.PdfParseStage`, runs the regex +
dictionary :class:`~packages.extractor.generic.GenericExtractor`
unconditionally, and optionally layers the Vietnamese precedent
normalizer :class:`~packages.extractor.precedent.PrecedentExtractor`
on top. Emits flat columns the downstream embedder / writer expects
(``text_hash``, ``char_len``, ``extracted``, ``precedent_number``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.extractor.generic import GenericExtractor
from packages.extractor.precedent import PrecedentExtractor


@dataclass
class LegalExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Run the generic (always) + precedent (optional) extractors."""

    cfg: Any
    name: str = "legal_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16

    _generic: GenericExtractor | None = field(default=None, init=False, repr=False)
    _precedent: PrecedentExtractor | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["markdown"])

    def outputs(self) -> tuple[list[str], list[str]]:
        # We always set the generic-layer columns. The precedent-layer
        # columns are also always emitted (None-valued when the site
        # layer is disabled) so schemas stay stable across sites.
        return (
            ["data"],
            [
                "text_hash",
                "char_len",
                "extracted",
                "precedent_number",
                "adopted_date",
                "applied_article_code",
                "applied_article_number",
                "applied_article_clause",
                "principle_text",
            ],
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._generic = GenericExtractor()
        self._precedent = PrecedentExtractor()

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._generic is None or self._precedent is None:
            self.setup(None)
        assert self._generic is not None and self._precedent is not None

        run_generic = bool(self.cfg.extractor.run_generic_layer)
        run_site = bool(self.cfg.extractor.run_site_layer)

        df = task.to_pandas().copy()

        text_hashes: list[str] = []
        char_lens: list[int] = []
        extracted_col: list[dict[str, Any]] = []
        precedent_numbers: list[str | None] = []
        adopted_dates: list[str | None] = []
        applied_codes: list[str | None] = []
        applied_numbers: list[int | None] = []
        applied_clauses: list[int | None] = []
        principle_texts: list[str | None] = []

        for _, row in df.iterrows():
            doc_id = str(row.get("doc_name") or row.get("doc_id") or "")
            markdown = str(row.get("markdown") or "")
            generic = self._generic.extract(doc_id=doc_id, markdown=markdown)

            text_hashes.append(generic.text_hash)
            char_lens.append(generic.char_len)
            extracted_col.append(
                generic.to_jsonable() if run_generic else {
                    "doc_id": doc_id,
                    "text_hash": generic.text_hash,
                    "char_len": generic.char_len,
                    "entities": [],
                    "relations": [],
                    "statute_refs": [],
                }
            )

            if run_site:
                precedent = self._precedent.extract(
                    doc_id=doc_id,
                    markdown=markdown,
                    scraper_metadata=_row_scraper_metadata(row),
                    generic=generic,
                )
                precedent_numbers.append(precedent.precedent_number)
                adopted_dates.append(precedent.adopted_date)
                applied_codes.append(precedent.applied_article_code)
                applied_numbers.append(precedent.applied_article_number)
                applied_clauses.append(precedent.applied_article_clause)
                principle_texts.append(precedent.principle_text)
            else:
                precedent_numbers.append(None)
                adopted_dates.append(None)
                applied_codes.append(None)
                applied_numbers.append(None)
                applied_clauses.append(None)
                principle_texts.append(None)

        df["text_hash"] = text_hashes
        df["char_len"] = char_lens
        df["extracted"] = extracted_col
        df["precedent_number"] = precedent_numbers
        df["adopted_date"] = adopted_dates
        df["applied_article_code"] = applied_codes
        df["applied_article_number"] = applied_numbers
        df["applied_article_clause"] = applied_clauses
        df["principle_text"] = principle_texts

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _row_scraper_metadata(row: Any) -> dict[str, Any]:
    """Collect scraper-supplied fields from a dataframe row for the precedent extractor."""
    keys = (
        "precedent_number",
        "adopted_date",
        "applied_article",
        "principle_text",
        "court",
        "source_judgment",
        "source_case",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in row and row[k] is not None:
            out[k] = row[k]
    return out


__all__ = ["LegalExtractStage"]
