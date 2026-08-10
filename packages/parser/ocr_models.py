"""Suggested OCR / VLM models for the deferred Vietnamese PDF cohort.

The triage pass never runs a vision model -- it only records *which*
model a deferred document should eventually go to. This module is that
recommendation table, keyed by :class:`VietnameseLegalPDFType`.

Every entry is ranked, and the ranking is anchored on measured
Vietnamese character error rate rather than general-purpose OCR
leaderboards. That distinction matters more than it sounds: Vietnamese
stacks up to three diacritics per vowel, and a recognizer trained on
generic Latin script drops them silently. The output still *looks* like
Vietnamese, so the failure is invisible to any length- or
language-detection based quality gate downstream.

Sources for the CER figures are recorded per-entry in
:attr:`SuggestedModel.evidence`. They come from published Vietnamese
OCR benchmarks, not from measurements taken on this corpus -- treat
them as a prior for picking the first model to try, then validate on a
held-out slice of the actual deferred cohort before committing to a
full run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.parser.types import OCR_TYPES, REPAIR_TYPES, VietnameseLegalPDFType

T = VietnameseLegalPDFType


@dataclass(frozen=True)
class SuggestedModel:
    """One ranked remediation option for a deferred document."""

    #: HuggingFace id, NIM slug, or CLI tool name.
    model_id: str
    #: How it runs: ``vllm`` | ``nim`` | ``transformers`` | ``cli``.
    backend: str
    #: ``self_hosted`` | ``cloud`` | ``local_cpu``.
    deployment: str
    #: Why this model is on the list at this rank.
    rationale: str
    #: Measured Vietnamese CER, where a published figure exists.
    evidence: str | None = None
    #: Rough operational cost: ``free`` | ``local_gpu`` | ``per_page``.
    cost_tier: str = "local_gpu"
    #: Set when the model is already wired into this repo, so using it
    #: needs no new infrastructure.
    already_deployed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Manifest-friendly plain dict (JSON-serialisable)."""
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "deployment": self.deployment,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "cost_tier": self.cost_tier,
            "already_deployed": self.already_deployed,
        }


# --------------------------------------------------------------- models

QWEN36_OMNI = SuggestedModel(
    model_id="Qwen/Qwen3.6-27B-FP8",
    backend="vllm",
    deployment="self_hosted",
    rationale=(
        "Already the production hybrid_fallback_runtime in this repo, so it "
        "needs no new infrastructure. Exposes parse_single_page, which is "
        "the only interface that supports per-page surgical OCR."
    ),
    evidence="in-repo A/B vs nemotron-omni, 2026-05 cutover",
    cost_tier="local_gpu",
    already_deployed=True,
)

VINTERN_1B = SuggestedModel(
    model_id="5CD-AI/Vintern-1B-v3_5",
    backend="transformers",
    deployment="self_hosted",
    rationale=(
        "Vietnamese-specialised VLM fine-tuned from InternVL2.5-1B. At ~1.8 GB "
        "it is the best diacritic fidelity per GPU-dollar on this list, and it "
        "is explicitly tuned for Vietnamese legal texts, invoices and tables. "
        "Pass whole pages, never line crops -- VLMs hallucinate at line scale."
    ),
    evidence="CER 0.47% clean / 0.37% noisy on nrl-ai/vn-synthetic-ocr (n=20 each)",
    cost_tier="local_gpu",
)

NEMOTRON_PARSE = SuggestedModel(
    model_id="nvidia/NVIDIA-Nemotron-Parse-v1.2",
    backend="vllm",
    deployment="self_hosted",
    rationale=(
        "The model behind NeMo Curator's own NemotronParsePDFReader, so the "
        "deferred manifest feeds it with no glue code. Uniquely returns layout "
        "bboxes and element classes (Section-header / Table / Page-footer), "
        "which the other options do not. Caveat: it is not Vietnamese-tuned, "
        "so validate diacritic fidelity on a sample before a full run."
    ),
    evidence="no published Vietnamese CER; layout quality is the draw, not text fidelity",
    cost_tier="local_gpu",
)

NEMOTRON_OMNI = SuggestedModel(
    model_id="nvidia/nemotron-3-nano-omni-30b",
    backend="vllm",
    deployment="self_hosted",
    rationale=(
        "Rollback target retained in this repo alongside Qwen3.6. Also exposes "
        "parse_single_page, so it is a valid surgical-OCR backend."
    ),
    evidence="in-repo baseline; dropped 7/20 pages on the largest reference PDF",
    cost_tier="local_gpu",
    already_deployed=True,
)

TESSERACT_VIE = SuggestedModel(
    model_id="tesseract-5 + vie traineddata",
    backend="cli",
    deployment="local_cpu",
    rationale=(
        "CPU-only baseline for the easy tail. Excellent on clean printed lines "
        "and effectively free, but degrades sharply on hard scans, so use it "
        "only after a quality gate has separated clean scans from poor ones."
    ),
    evidence="CER 0.00% clean printed / 0.70% noisy / 30.34% hard scan",
    cost_tier="free",
)

VIETOCR = SuggestedModel(
    model_id="pbcquoc/vietocr (vgg_transformer)",
    backend="transformers",
    deployment="self_hosted",
    rationale=(
        "Vietnamese-specific handwriting recogniser, for the handwritten "
        "annotations and signatures that appear on scanned judgments. Needs a "
        "separate text detector (CRAFT or DB) to produce line crops first."
    ),
    evidence="CER 31.82% handwritten vs Tesseract 69.34%",
    cost_tier="local_gpu",
)

QPDF_DECRYPT = SuggestedModel(
    model_id="qpdf --decrypt",
    backend="cli",
    deployment="local_cpu",
    rationale=(
        "Not an OCR model. Strips the encryption layer so the file can be "
        "re-triaged; most Vietnamese court PDFs carry owner-password "
        "restrictions with an empty user password, which qpdf removes "
        "without a credential."
    ),
    cost_tier="free",
)

MUTOOL_CLEAN = SuggestedModel(
    model_id="mutool clean -gggg",
    backend="cli",
    deployment="local_cpu",
    rationale=(
        "Not an OCR model. Rebuilds broken xref tables and truncated object "
        "streams so the file can be re-triaged. Try Ghostscript "
        "(gs -o out.pdf -sDEVICE=pdfwrite) as a second pass if mutool fails."
    ),
    cost_tier="free",
)


#: Models explicitly rejected, recorded so the decision is not silently
#: relitigated the next time someone reaches for a familiar OCR stack.
NOT_RECOMMENDED: list[dict[str, str]] = [
    {
        "model_id": "PaddleOCR PP-OCRv5 (lang='vi')",
        "reason": (
            "Ships no Vietnamese-specific recogniser. Setting lang='vi' silently "
            "loads the generic latin_PP-OCRv5_mobile_rec model, which strips "
            "diacritics and produces text that still looks Vietnamese but is "
            "wrong -- the worst possible failure mode for a legal corpus."
        ),
        "evidence": "CER 24.70% clean printed / 31.33% noisy / 86.13% hard scan",
    },
    {
        "model_id": "RapidOCR (ONNX port of PaddleOCR)",
        "reason": "Detector fails to find Vietnamese text lines on most scans.",
        "evidence": "CER 63.97% clean printed / 100% hard scan",
    },
    {
        "model_id": "TrOCR-handwritten",
        "reason": "English-only training; no Vietnamese diacritic coverage.",
        "evidence": "CER 75.89% on Vietnamese handwriting",
    },
]


#: Ranked remediation per deferred type. Native types map to an empty
#: list -- they never reach the deferred manifest.
SUGGESTED_MODELS: dict[VietnameseLegalPDFType, list[SuggestedModel]] = {
    # Image-only scans: whole-page VLM OCR. Qwen leads on
    # already-deployed, Vintern leads on measured Vietnamese accuracy.
    T.SCANNED_IMAGE: [QWEN36_OMNI, VINTERN_1B, NEMOTRON_PARSE, TESSERACT_VIE, VIETOCR],
    # Font-corrupted: the text layer exists but lies, so it must be
    # ignored entirely and the page re-read from pixels. Same cohort as
    # a scan, minus the handwriting recogniser -- these are digital
    # documents with broken fonts, never handwritten.
    T.FONT_CORRUPTED: [VINTERN_1B, QWEN36_OMNI, NEMOTRON_PARSE, TESSERACT_VIE],
    # Mixed pages: only the empty pages need OCR, so the backend must
    # expose parse_single_page. That requirement excludes Vintern and
    # Tesseract from the top ranks here.
    T.MIXED_PAGES: [QWEN36_OMNI, NEMOTRON_OMNI, VINTERN_1B],
    # Repair cases: no model applies until the bytes are fixed.
    T.ENCRYPTED: [QPDF_DECRYPT],
    T.CORRUPTED: [MUTOOL_CLEAN],
}


def suggested_models(pdf_type: VietnameseLegalPDFType) -> list[SuggestedModel]:
    """Ranked remediation options for ``pdf_type`` (empty when native)."""
    return SUGGESTED_MODELS.get(pdf_type, [])


def suggested_models_as_dicts(
    pdf_type: VietnameseLegalPDFType,
) -> list[dict[str, Any]]:
    """:func:`suggested_models` flattened for JSONL serialisation."""
    return [m.to_dict() for m in suggested_models(pdf_type)]


def primary_model(pdf_type: VietnameseLegalPDFType) -> str | None:
    """Top-ranked ``model_id`` for ``pdf_type``, or ``None`` when native."""
    ranked = suggested_models(pdf_type)
    return ranked[0].model_id if ranked else None


def _assert_catalog_complete() -> None:
    """Every deferred type must carry at least one suggestion.

    Guards against adding a :class:`VietnameseLegalPDFType` member and
    forgetting the catalog entry, which would silently emit deferred
    manifest rows with no remediation path.
    """
    missing = sorted(
        t.value for t in (OCR_TYPES | REPAIR_TYPES) if not SUGGESTED_MODELS.get(t)
    )
    if missing:
        msg = f"SUGGESTED_MODELS is missing entries for deferred types: {missing}"
        raise RuntimeError(msg)


_assert_catalog_complete()


__all__ = [
    "NOT_RECOMMENDED",
    "SUGGESTED_MODELS",
    "SuggestedModel",
    "primary_model",
    "suggested_models",
    "suggested_models_as_dicts",
]
