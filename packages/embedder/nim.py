"""NIM (OpenAI-compatible) embedding backend."""

from __future__ import annotations

from packages.embedder.base import EmbedderBackend


class NimEmbedder(EmbedderBackend):
    """NVIDIA NIM OpenAI-compatible ``/v1/embeddings`` endpoint.

    Embedding dim is probed lazily on the first call unless provided
    via ``embedding_dim`` (registry hint avoids the probe).
    """

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        embedding_dim: int | None = None,
        max_seq_length: int = 512,
    ) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self._embedding_dim_hint = embedding_dim
        self.max_seq_length = max_seq_length
        self._embedding_dim: int | None = embedding_dim

    @property
    def embedding_dim(self) -> int:  # type: ignore[override]
        if self._embedding_dim is None:
            # Probe once with a trivial input.
            vec = self.embed_batch(["probe"])[0]
            self._embedding_dim = len(vec)
        return self._embedding_dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # NIM embedding models require ``input_type``: "query" for
        # search queries, "passage" for documents being indexed. ViLA
        # embeds full precedent markdown bodies for retrieval, so
        # "passage" is correct. Passed via ``extra_body`` for SDK
        # pass-through.
        resp = self._client.embeddings.create(
            model=self.model_id,
            input=texts,
            encoding_format="float",
            extra_body={"input_type": "passage"},
        )
        return [list(r.embedding) for r in resp.data]


__all__ = ["NimEmbedder"]
