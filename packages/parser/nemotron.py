"""NVIDIA ``nemotron-parse`` NIM parser backend.

Posts each document to the NIM ``/cv/nvidia/nemotron-parse`` endpoint
and normalizes the response into the dict shape declared in
:class:`packages.parser.base.ParserAlgorithm`.

Uses direct :mod:`requests` rather than the OpenAI SDK because
nemotron-parse is a CV / document endpoint under
``ai.api.nvidia.com/v1/cv/``, not a chat/completions model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)


class NemotronParser(ParserAlgorithm):
    """Thin wrapper over the NIM ``nemotron-parse`` endpoint."""

    runtime = "nim"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://ai.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-parse",
        timeout: float = 120.0,
    ) -> None:
        import requests

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        self._base_url = base_url.rstrip("/")
        self.model_id = model
        self._timeout = timeout

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        import base64

        payload_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        url = f"{self._base_url}/cv/{self.model_id}"
        resp = self._session.post(
            url,
            json={
                "input": [
                    {"type": "file", "data": payload_b64, "mime_type": "application/pdf"}
                ],
                "options": {
                    "preserve_tables": preserve_tables,
                    "emit_layout": True,
                },
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return _normalize_nemotron_response(resp.json())


def _normalize_nemotron_response(resp: Any) -> dict[str, Any]:
    """Coerce a nemotron-parse response into the shape ViLA expects."""
    if isinstance(resp, dict):
        return resp
    # SDK may return a typed object; best-effort conversion.
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))


__all__ = ["NemotronParser"]
