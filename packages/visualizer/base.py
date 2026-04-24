"""Abstract :class:`Renderer` base + pipeline-output loader.

The visualizer is no longer a pipeline stage. Every :class:`Renderer`
takes a pandas :class:`DataFrame` and writes one or more files under
``out_dir``. :func:`load_pipeline_output` reads the two output trees
the four-pipeline chain leaves on disk:

    ``<host>/jsonl/*.jsonl``             (Extractor output: text + entities)
    ``<host>/parquet/reduced/*.parquet`` (Reducer output: embeddings + coords)

and joins them on ``doc_name``, then applies ontology-aligned fill
columns. Either directory may be absent (partial runs), in which case
the other is returned alone.

Invoke the whole renderer bundle from the command line via
``python -m apps.visualizer``.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology, arc_for_code_id

logger = logging.getLogger(__name__)


class Renderer(abc.ABC):
    """One visualization artifact (HTML file, notebook, ...)."""

    #: Short slug used in logs and ``counts`` buckets.
    name: str = ""

    #: Which bucket in the app's counts dict this Renderer bumps on
    #: success (``"scatters"``, ``"distributions"``, ``"misc"``, ...).
    bucket: str = "misc"

    @abc.abstractmethod
    def render(
        self,
        df: pd.DataFrame,
        *,
        out_dir: Path,
        cfg: Any,
        onto: Ontology,
        slug: str,
        force: bool,
    ) -> int:
        """Render the artifact(s). Return the number of files written."""


# ------------------------------------------------------ pipeline-output loader


def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    if not parquet_dir.exists():
        return pd.DataFrame()
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _read_jsonl_dir(jsonl_dir: Path) -> pd.DataFrame:
    if not jsonl_dir.exists():
        return pd.DataFrame()
    files = sorted(jsonl_dir.glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_json(p, lines=True) for p in files if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_pipeline_output(
    parquet_dir: Path,
    jsonl_dir: Path | None = None,
) -> pd.DataFrame:
    """Read the pipeline's dual-writer output and join on ``doc_name``.

    * ``<parquet_dir>/*.parquet`` carries the embeddings + reducer
      coords + ``doc_name``.
    * ``<jsonl_dir>/*.jsonl`` carries the text + extracted entities +
      precedent metadata.

    Either directory may be missing. When ``jsonl_dir`` is ``None``
    the loader falls back to reading parquet only (preserves the
    pre-dual-writer behaviour).
    """
    pq = _read_parquet_dir(Path(parquet_dir))
    if jsonl_dir is None:
        return pq
    jl = _read_jsonl_dir(Path(jsonl_dir))
    if pq.empty:
        return jl
    if jl.empty:
        return pq
    # doc_name is the join key produced by every stage. Parquet
    # columns win on conflict (the vectors + coords are authoritative
    # for the fields they own).
    return jl.merge(pq, on="doc_name", how="outer", suffixes=("_jsonl", ""))


def apply_ontology(df: pd.DataFrame, onto: Ontology) -> pd.DataFrame:
    """Fill ontology-aligned columns renderers expect.

    Defaults ``legal_type``, ``legal_relation``, ``procedure_type``,
    ``code_id``, ``legal_arc``, ``cluster_id`` on the incoming frame
    (any column already populated by the pipeline is left alone).
    Returns the same frame for chaining.
    """
    if df.empty:
        return df
    if "doc_id" not in df.columns and "doc_name" in df.columns:
        df["doc_id"] = df["doc_name"]
    if "legal_type" not in df.columns:
        df["legal_type"] = "precedent"
    if "legal_relation" not in df.columns:
        df["legal_relation"] = "(unknown)"
    df["legal_relation"] = df["legal_relation"].map(
        lambda v: onto.normalize_enum("LegalRelation", v)
    )
    if "procedure_type" not in df.columns:
        df["procedure_type"] = "(unknown)"
    df["procedure_type"] = df["procedure_type"].map(
        lambda v: onto.normalize_enum("ProcedureType", v)
    )
    if "applied_article_code" in df.columns:
        df["code_id"] = df["applied_article_code"].fillna("(unknown)")
    elif "code_id" not in df.columns:
        df["code_id"] = "(unknown)"
    df["legal_arc"] = df["code_id"].map(
        lambda c: (arc_for_code_id(c).id if arc_for_code_id(c) else "(unknown)")
    )
    if "cluster_id" not in df.columns:
        df["cluster_id"] = -1
    return df


def build_dataset(
    parquet_dir: Path,
    onto: Ontology,
    jsonl_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the pipeline's dual-writer output + fill ontology columns."""
    df = load_pipeline_output(parquet_dir, jsonl_dir=jsonl_dir)
    return apply_ontology(df, onto)


__all__ = [
    "Renderer",
    "apply_ontology",
    "build_dataset",
    "load_pipeline_output",
]
