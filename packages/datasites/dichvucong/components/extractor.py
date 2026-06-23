"""dichvucong DocumentExtractor: raw API record -> curated row.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentExtractor`.
Flattens one ``procedure_advanced_search_service_v2`` record into the
stable, English-snake_case row schema the writer + downstream stages
expect, and stamps the two freshness keys used by the incremental
mechanism (``wiki/DICHVUCONG.md`` §5):

* ``decision_id`` (``QDCBID``) — the công-bố decision that issued /
  amended this procedure. A change here means supersession.
* ``content_hash`` — sha1 over the salient fields; the exact-change key.

``doc_name`` is the ``PROCEDURE_CODE`` (e.g. ``1.015045``) — the
canonical, human-facing citation and a stable per-procedure key.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from nemo_curator.stages.text.download.base import DocumentExtractor

_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _slug(code: str, procedure_id: str) -> str:
    base = _SLUG_RE.sub("_", code).strip("_")
    return base or f"id_{procedure_id}"


class DichvucongDocumentExtractor(DocumentExtractor):
    """Project one raw API record into the curated procedure row."""

    def __init__(self, cfg: Any) -> None:
        self._host = str(cfg.host)
        self._detail_tpl = str(
            cfg.scraper.get(
                "detail_url_template",
                "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc={code}",
            )
        )

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        r = record.get("record") or {}
        procedure_id = _s(r.get("ID"))
        code = _s(r.get("PROCEDURE_CODE"))
        if not (procedure_id or code):
            return None
        name = _s(r.get("PROCEDURE_NAME"))
        decision_id = _s(r.get("QDCBID"))

        content_hash = hashlib.sha1(
            "|".join(
                [
                    code,
                    name,
                    _s(r.get("PUBLISHED_AGENCY")),
                    _s(r.get("IMPLEMENTATION_AGENCY")),
                    _s(r.get("FIELD_NAME")),
                    decision_id,
                ]
            ).encode("utf-8")
        ).hexdigest()

        return {
            "doc_name": _slug(code, procedure_id),
            "procedure_id": procedure_id,
            "procedure_code": code,
            "procedure_name": name,
            "published_agency": _s(r.get("PUBLISHED_AGENCY")),
            "implementation_agency": _s(r.get("IMPLEMENTATION_AGENCY")),
            "field_name": _s(r.get("FIELD_NAME")),
            "decision_id": decision_id,
            "amount": _s(r.get("AMOUNT")),
            "source": self._host,
            "source_url": self._detail_tpl.format(code=code, id=procedure_id),
            "content_hash": content_hash,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def input_columns(self) -> list[str]:
        return ["page_path", "record"]

    def output_columns(self) -> list[str]:
        return [
            "doc_name",
            "procedure_id",
            "procedure_code",
            "procedure_name",
            "published_agency",
            "implementation_agency",
            "field_name",
            "decision_id",
            "amount",
            "source",
            "source_url",
            "content_hash",
            "fetched_at",
        ]


__all__ = ["DichvucongDocumentExtractor"]
