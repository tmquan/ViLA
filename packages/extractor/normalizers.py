"""Declarative normalizer registry + Curator chain stage (wiki/DATASITES.md §3.5).

The two-tier output rule + the user directive that "any normalization
should be a list and part of nemo curator for reproducibility" boil
down to:

1. A site declares its normalization recipe as
   ``cfg.extractor.normalizers: list[str]`` — each entry is the
   registered name of a normalizer.
2. Names resolve via :data:`NORMALIZER_REGISTRY` to
   :class:`Normalizer` instances. Each one mutates one or more
   columns of a pandas ``DataFrame`` row-wise.
3. :class:`NormalizerChainStage` is a Curator
   :class:`~nemo_curator.stages.base.ProcessingStage` that runs the
   resolved chain on a :class:`~nemo_curator.tasks.DocumentBatch`.
   It sits between the reader and
   :class:`~packages.extractor.stage.LegalExtractStage`.
4. The resolved chain shows up in ``Pipeline.describe()`` and is
   recorded in ``manifest.json`` so the corpus is reproducible from
   the config alone.

The built-in registration only covers the cross-corpus
``vietnamese_text`` normalizer (ftfy + NFC + tone-mark + whitespace).
Site-specific normalizers register themselves on import — see
:mod:`packages.datasites.vbpl.normalizers` for an example.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import pandas as pd

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.extractor.normalization import normalize_text

logger = logging.getLogger(__name__)


@runtime_checkable
class Normalizer(Protocol):
    """Row-wise normalizer that mutates a ``pandas.DataFrame`` in place.

    Each entry in :data:`NORMALIZER_REGISTRY` implements this protocol.
    The chain stage applies them in order to the batch's DataFrame.

    Attributes
    ----------
    name : str
        Stable identifier used in ``cfg.extractor.normalizers`` and
        in ``manifest.json``.
    columns : tuple[str, ...]
        Output columns this normalizer writes. Used by the chain stage
        to know which columns are touched (for schema-validation
        hints + descriptive logging). Empty tuple means "no column
        guaranteed" (e.g. side-effect-only normalizer).
    """

    name: str
    columns: tuple[str, ...]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with this normalizer's column(s) updated.

        The implementation may mutate ``df`` in place and return it,
        or return a new DataFrame. The chain stage uses whichever is
        returned.
        """
        ...


#: Global registry mapping ``name`` -> :class:`Normalizer` instance.
#: Populated by :func:`register_normalizer` decorators. Site packages
#: import this module and register their normalizers at import time.
NORMALIZER_REGISTRY: dict[str, Normalizer] = {}


def register_normalizer(
    name: str | None = None,
) -> Callable[[type[Normalizer]], type[Normalizer]]:
    """Class decorator that registers a normalizer in the global registry.

    Usage::

        @register_normalizer("vietnamese_text")
        class VietnameseTextNormalizer:
            name = "vietnamese_text"
            columns = ("markdown",)
            def apply(self, df): ...

    The decorator instantiates the class once (normalizers are
    stateless) and stores the singleton under ``name`` (or
    ``cls.name`` when ``name`` is None).
    """

    def decorator(cls: type[Normalizer]) -> type[Normalizer]:
        instance = cls()
        key = name or getattr(instance, "name", cls.__name__)
        if not key:
            raise ValueError(
                f"normalizer {cls.__name__} has no 'name' attribute"
            )
        if key in NORMALIZER_REGISTRY:
            existing = type(NORMALIZER_REGISTRY[key]).__name__
            if existing != cls.__name__:
                logger.warning(
                    "normalizer name %r already registered by %s; "
                    "overriding with %s",
                    key, existing, cls.__name__,
                )
        NORMALIZER_REGISTRY[key] = instance
        return cls

    return decorator


def resolve_normalizer_names(cfg: Any) -> list[str]:
    """Return the resolved normalizer list from ``cfg.extractor``.

    Resolution order:

    1. ``cfg.extractor.normalizers`` -- the canonical declarative
       list. When present (including the empty list), this is the
       full answer. An empty list explicitly disables normalization.
    2. Legacy ``run_text_normalization`` boolean -- only consulted
       when ``normalizers`` is missing entirely from cfg (older
       schema). Emits a deprecation warning on every hit so sites
       still on the legacy knob get a nudge to migrate. ``true``
       maps to ``["vietnamese_text"]``; ``false`` (or missing) maps
       to ``[]`` (no normalization).
    """
    if cfg is None:
        return []
    extractor_cfg = _safe_attr(cfg, "extractor")
    if extractor_cfg is None:
        return []

    names = _safe_attr(extractor_cfg, "normalizers")
    if names is not None:
        try:
            return [str(n) for n in names if n]
        except TypeError:
            logger.warning(
                "cfg.extractor.normalizers=%r is not iterable; "
                "treating as empty", names,
            )
            return []

    legacy = _safe_attr(extractor_cfg, "run_text_normalization")
    if legacy is None:
        return []
    logger.warning(
        "cfg.extractor.run_text_normalization=%r is deprecated; "
        "declare `extractor.normalizers: [vietnamese_text]` in the "
        "site config instead. Honoured for backward-compat in this "
        "run.", legacy,
    )
    return ["vietnamese_text"] if bool(legacy) else []


def build_normalizer_chain(cfg: Any) -> "NormalizerChainStage | None":
    """Return a chain stage from ``cfg``, or ``None`` if the chain is empty.

    ``cfg.extractor.normalizer_fail_fast`` (optional, default False)
    promotes per-step exceptions from warning-and-skip to hard abort.
    """
    names = resolve_normalizer_names(cfg)
    if not names:
        return None
    fail_fast = False
    extractor_cfg = _safe_attr(cfg, "extractor")
    if extractor_cfg is not None:
        ff = _safe_attr(extractor_cfg, "normalizer_fail_fast")
        if ff is not None:
            fail_fast = bool(ff)
    return NormalizerChainStage(normalizers=tuple(names), fail_fast=fail_fast)


# --------------------------------------------------------------------- chain stage


@dataclass
class NormalizerChainStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Curator stage that applies a resolved list of normalizers in order.

    Inputs the row's ``doc_name`` + any columns the chain's
    normalizers touch (resolved via the registry); outputs the same
    columns post-normalization.

    The stage's ``name`` reflects the resolved chain so
    ``Pipeline.describe()`` and the manifest show
    ``normalizer_chain[vietnamese_text,strip_markdown_junk,...]``.
    """

    normalizers: tuple[str, ...]
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16
    name: str = field(init=False, default="normalizer_chain")
    # If True, an exception from any normalizer aborts the stage
    # instead of silently dropping that normalizer's contribution.
    # Defaults to False to preserve the historical "best-effort"
    # contract; flip on for runs where partial normalization would
    # poison downstream extractors.
    fail_fast: bool = False

    _resolved: tuple[Normalizer, ...] = field(
        default_factory=tuple, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self._resolved = _resolve(self.normalizers)
        if self._resolved:
            self.name = "normalizer_chain[" + ",".join(
                n.name for n in self._resolved
            ) + "]"

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["doc_name"])

    def outputs(self) -> tuple[list[str], list[str]]:
        cols: list[str] = []
        for n in self._resolved:
            for c in getattr(n, "columns", ()):
                if c and c not in cols:
                    cols.append(c)
        return (["data"], cols)

    def setup(
        self, worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # Normalizers are stateless instances cached in the registry;
        # nothing to set up per-worker. Re-resolving here lets a
        # downstream worker pick up newly-registered normalizers
        # (matters when the chain is built on the driver and the
        # registry is repopulated on the worker via pickle).
        self._resolved = _resolve(self.normalizers)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if not self._resolved:
            return task
        df = task.to_pandas().copy()
        for normalizer in self._resolved:
            try:
                df = normalizer.apply(df)
            except Exception:
                if self.fail_fast:
                    logger.exception(
                        "normalizer %s raised on task %s; aborting (fail_fast)",
                        normalizer.name, task.task_id,
                    )
                    raise
                logger.exception(
                    "normalizer %s raised on task %s; skipping",
                    normalizer.name, task.task_id,
                )
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _resolve(names: Sequence[str]) -> tuple[Normalizer, ...]:
    """Look up each name in the registry; warn + skip on misses."""
    out: list[Normalizer] = []
    for n in names:
        if not n:
            continue
        normalizer = NORMALIZER_REGISTRY.get(n)
        if normalizer is None:
            logger.warning(
                "normalizer %r not registered; skipping. "
                "Available: %s", n, sorted(NORMALIZER_REGISTRY),
            )
            continue
        out.append(normalizer)
    return tuple(out)


def _safe_attr(obj: Any, key: str) -> Any:
    """Best-effort attribute / item lookup that never raises."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except Exception:  # noqa: BLE001
            pass
    try:
        return getattr(obj, key)
    except AttributeError:
        return None


# --------------------------------------------------------------------- built-ins


@register_normalizer("vietnamese_text")
class VietnameseTextNormalizer:
    """ftfy + NFC + Vietnamese tone-mark + PDF whitespace, on ``markdown``.

    Cross-corpus default. Wraps the existing
    :func:`packages.extractor.normalization.normalize_text` pure
    function so the chain shares the same canonical implementation
    every legacy ``LegalExtractStage`` already used.
    """

    name: str = "vietnamese_text"
    columns: tuple[str, ...] = ("markdown",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns:
            return df
        df["markdown"] = df["markdown"].map(
            lambda v: normalize_text(v) if isinstance(v, str) and v else v,
        )
        return df


__all__ = [
    "NORMALIZER_REGISTRY",
    "Normalizer",
    "NormalizerChainStage",
    "VietnameseTextNormalizer",
    "build_normalizer_chain",
    "register_normalizer",
    "resolve_normalizer_names",
]
