"""Text chunking helpers used when the embedder's window < full doc.

Pipeline-wide ``full_text_context`` is 32k tokens; most embedding
models have smaller windows (e.g. 8k for
``nvidia/llama-nemotron-embed-1b-v2``). When the doc exceeds the
window, the stage splits into chunks, embeds each, and mean-pools
the vectors to preserve the full 32k of context in one doc-level
vector.

Three modes:
    off       -> return the text as-is (backend truncates)
    sliding   -> character-approximate sliding window with overlap
    sentence  -> sentence-boundary split with a soft cap
"""

from __future__ import annotations

import math
import re


def chunk_sliding(text: str, window: int, overlap: int) -> list[str]:
    """Character-approximate sliding window (token-proxy)."""
    if len(text) <= window:
        return [text]
    step = max(1, window - overlap)
    return [text[i : i + window] for i in range(0, len(text), step) if i < len(text)]


def chunk_sentence(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Sentence-boundary chunking with a soft cap.

    Splits on Western and Vietnamese sentence terminators; carries an
    ``overlap_chars`` tail from one chunk to the next so sentence
    boundaries don't cost context at the split.
    """
    parts = re.split(r"(?<=[.!?\u3002])\s+", text)  # incl. Vietnamese period
    chunks: list[str] = []
    buf = ""
    for sent in parts:
        if len(buf) + len(sent) + 1 > target_chars and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap_chars:] + " " + sent if overlap_chars > 0 else sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        chunks.append(buf.strip())
    return chunks or [text]


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """Mean-pool a list of equal-length vectors, then L2-normalize."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return list(vectors[0])

    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValueError(
                f"inconsistent embedding dim across chunks: expected {dim}, got {len(v)}"
            )
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(vectors))
    pooled = [x / n for x in acc]
    norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
    return [x / norm for x in pooled]


__all__ = ["chunk_sentence", "chunk_sliding", "mean_pool"]
