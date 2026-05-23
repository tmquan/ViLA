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

import math
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.extractor.generic import GenericExtractor
from packages.extractor.normalization import normalize_text
from packages.extractor.precedent import PrecedentExtractor
from packages.extractor.structure import LegalStructureExtractor


@dataclass
class LegalExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Run the generic (always) + precedent (optional) + structure layers."""

    cfg: Any
    name: str = "legal_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16

    _generic: GenericExtractor | None = field(default=None, init=False, repr=False)
    _precedent: PrecedentExtractor | None = field(default=None, init=False, repr=False)
    _structure: LegalStructureExtractor | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["markdown"])

    def outputs(self) -> tuple[list[str], list[str]]:
        # We always set the generic-layer columns. The precedent-layer
        # and structure-layer columns are also always emitted
        # (None-valued when their layer is disabled) so schemas stay
        # stable across sites.
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
                "structure",
            ],
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._generic = GenericExtractor()
        self._precedent = PrecedentExtractor()
        self._structure = LegalStructureExtractor()

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if (
            self._generic is None
            or self._precedent is None
            or self._structure is None
        ):
            self.setup(None)
        assert (
            self._generic is not None
            and self._precedent is not None
            and self._structure is not None
        )

        run_generic = bool(self.cfg.extractor.run_generic_layer)
        run_site = bool(self.cfg.extractor.run_site_layer)
        run_structure = bool(
            getattr(self.cfg.extractor, "run_structure_layer", True)
        )
        # Skip inline normalization when an upstream
        # :class:`NormalizerChainStage` already ran (the new
        # declarative chain owns normalization end-to-end —
        # wiki/DATASITES.md §3.5). The legacy ``run_text_normalization``
        # boolean keeps the inline path alive for sites that
        # haven't migrated to the chain yet.
        chain_names = getattr(self.cfg.extractor, "normalizers", None) or []
        run_normalize = bool(
            getattr(self.cfg.extractor, "run_text_normalization", True)
        ) and not chain_names

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
        structures: list[dict[str, Any] | None] = []
        normalized_markdowns: list[str] = []

        for _, row in df.iterrows():
            doc_id = str(row.get("doc_name") or row.get("doc_id") or "")
            markdown = str(row.get("markdown") or "")
            # Canonicalize Unicode + Vietnamese tone-mark orthography +
            # PDF whitespace artefacts before any layer inspects the
            # text. Lets every regex in the layers below assume a
            # single canonical form (NFC, modern orthography). The
            # normalized markdown overwrites the input column so
            # char-spans in `extracted` and `structure` index into
            # the same string that lands in JSONL.
            if run_normalize and markdown:
                markdown = normalize_text(markdown)
            normalized_markdowns.append(markdown)
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

            if run_structure:
                structure = self._structure.extract(
                    doc_id=doc_id,
                    markdown=markdown,
                    scraper_metadata=_row_scraper_metadata(row),
                )
                structures.append(structure.to_jsonable())
            else:
                structures.append(None)

        if run_normalize:
            df["markdown"] = normalized_markdowns
        df["text_hash"] = text_hashes
        df["char_len"] = char_lens
        df["extracted"] = extracted_col
        df["precedent_number"] = precedent_numbers
        df["adopted_date"] = adopted_dates
        df["applied_article_code"] = applied_codes
        df["applied_article_number"] = applied_numbers
        df["applied_article_clause"] = applied_clauses
        df["principle_text"] = principle_texts
        df["structure"] = structures

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _row_scraper_metadata(row: Any) -> dict[str, Any]:
    """Collect scraper-supplied fields from a dataframe row.

    Read by the precedent extractor (legal-citation normalisation) and
    by the structure extractor (header-meta hints when the parser
    output is too noisy to recover them from the markdown alone).
    """
    keys = (
        "precedent_number",
        "adopted_date",
        "applied_article",
        "principle_text",
        "court",
        "source_judgment",
        "source_case",
        "title",
        "doc_type",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in row:
            continue
        v = row[k]
        if v is None:
            continue
        # pandas surfaces missing values as ``float('nan')`` which
        # is truthy in ``or "" `` short-circuits and crashes
        # downstream ``str``-only consumers; skip those exactly
        # like None.
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = v
    return out


__all__ = ["LegalExtractStage"]
