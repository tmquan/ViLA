"""Local HuggingFace embedder backend.

Mean-pools the last hidden state by default. For models with a
dedicated pooling layer (e.g. sentence-transformers), users can point
``model_id`` at a sentence-transformers repo and this wrapper will
still function (mean pooling is safe across both).
"""

from __future__ import annotations

import logging

from packages.embedder.base import EmbedderBackend

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(EmbedderBackend):
    """Local ``transformers`` embedder with mean-pool + L2-normalize."""

    def __init__(
        self,
        model_id: str,
        *,
        max_seq_length: int,
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        resolved_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype)

        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(resolved_device)
        self._model.eval()
        self._device = resolved_device

        # Resolve max_seq_length against the model's capability.
        model_max = int(
            getattr(self._model.config, "max_position_embeddings", max_seq_length)
        )
        if max_seq_length > model_max:
            logger.warning(
                "requested max_seq_length=%d exceeds model's max_position_embeddings=%d; "
                "clamping (or use chunking: sliding to preserve full context)",
                max_seq_length,
                model_max,
            )
        self.max_seq_length = min(max_seq_length, model_max)

        # Probe embedding dim from hidden size.
        self.embedding_dim = int(self._model.config.hidden_size)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import torch

        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**enc)
        # Mean-pool over the attention mask.
        last = out.last_hidden_state                          # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)  # (B, T, 1)
        summed = (last * mask).sum(dim=1)                     # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)
        vecs = summed / counts
        # Unit-normalize.
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return [v.detach().cpu().float().tolist() for v in vecs]


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.float32)


__all__ = ["HuggingFaceEmbedder"]
