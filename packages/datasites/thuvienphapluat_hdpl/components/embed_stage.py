"""NeMo Curator stage: embed a hoi-dap Q&A batch (question + answer).

Asymmetric retrieval against the local vLLM ``nvidia/Nemotron-3-Embed-8B-BF16``
OpenAI ``/v1/embeddings`` endpoint: the question gets the model's ``query: ``
prompt, the answer gets ``passage: `` (a question vector then retrieves its
answer vector). Both 4096-d, L2-normalised by the model. A :class:`DocumentBatch`
with ``id``/``question``/``answer`` gains ``question_embedding`` +
``answer_embedding``. Driven in-process on the GB10, sharing the vLLM server with
the other embed jobs.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

logger = logging.getLogger(__name__)


@dataclass
class TVPLQAEmbedStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Embed each record's question (``query: ``) + answer (``passage: ``)."""

    base_url: str = "http://localhost:8000/v1"
    model_id: str = "nvidia/Nemotron-3-Embed-8B-BF16"
    hard_chars: int = 30000        # truncate a lone over-long text (< max-model-len)
    req_texts: int = 32            # texts per HTTP request
    concurrency: int = 6           # in-flight requests (shares the vLLM server)
    name: str = "tvpl_qa_embed"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["id", "question", "answer"])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["id", "question_embedding", "answer_embedding",
                           "embedding_dim", "embedding_model_id"])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        pass

    def _raw_post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_id, "input": texts}).encode()
        req = urllib.request.Request(f"{self.base_url}/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]

    def _post(self, texts: list[str]) -> list[list[float]]:
        """Transient-retry + 400-aware bisect/truncate (output stays 1:1)."""
        for attempt in range(4):
            try:
                return self._raw_post(texts)
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    if len(texts) == 1:
                        if len(texts[0]) > self.hard_chars:
                            return self._raw_post([texts[0][:self.hard_chars]])
                        raise
                    mid = len(texts) // 2
                    return self._post(texts[:mid]) + self._post(texts[mid:])
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        return []

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas()
        q_texts = ["query: " + (str(x) if x is not None else "")[:self.hard_chars]
                   for x in df["question"]]
        a_texts = ["passage: " + (str(x) if x is not None else "")[:self.hard_chars]
                   for x in df["answer"]]
        flat = q_texts + a_texts
        reqs = [flat[i:i + self.req_texts] for i in range(0, len(flat), self.req_texts)]
        vecs: list[list[float]] = []
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for part in ex.map(self._post, reqs):
                vecs.extend(part)
        n = len(df)
        q_vecs, a_vecs = vecs[:n], vecs[n:]
        out = pd.DataFrame({
            "id": df["id"].astype(str).tolist(),
            "question_embedding": q_vecs,
            "answer_embedding": a_vecs,
            "embedding_dim": [len(q_vecs[0]) if q_vecs else 0] * n,
            "embedding_model_id": [self.model_id] * n,
        })
        return DocumentBatch(task_id=task.task_id, dataset_name=task.dataset_name, data=out)


__all__ = ["TVPLQAEmbedStage"]
