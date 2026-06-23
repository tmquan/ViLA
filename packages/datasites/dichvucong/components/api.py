"""Shared client for the national public-service portal gateway.

Every data call on ``dichvucong.gov.vn`` (Cổng Dịch vụ công Quốc gia,
run by Văn phòng Chính phủ — "VPCP") funnels through one JSP gateway::

    POST https://vpcp.dichvucong.gov.vn/jsp/rest.jsp
    Content-Type: application/x-www-form-urlencoded
    body: params=<URL-encoded JSON>

The JSON ``params`` object selects one of two query families:

* ``type="ref"`` + ``service="<name>"`` + ``provider="dvcquocgia"`` —
  the relational reference services (search, agency/field/object lists,
  detail). Returns a JSON array of records.
* ``type="fts"`` + ``source_data="thu_tuc_v1"`` + ``key_search`` — a
  Solr full-text query. Returns a Solr envelope
  (``{"response": {"numFound": N, "docs": [...]}}``).

This module hides the gateway shape behind two helpers
(:func:`call_ref`, :func:`call_fts`) and a tiny page-locator codec
(:func:`encode_page_url` / :func:`decode_page_url`) so Curator's
URL-string contract can carry the ``(agency, page)`` coordinates a
``rest.jsp`` POST needs.

The single confirmed-live national source already aggregates **every
ministry and province** (a search with no agency filter returns Bộ
Công An, Bộ Tài chính, … rows), so one ``dichvucong`` datasite covers
``bocongan`` and the other sub-portals — see ``wiki/DICHVUCONG.md``.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlencode

from packages.common.http import PoliteSession

logger = logging.getLogger(__name__)

DEFAULT_REST_URL = "https://vpcp.dichvucong.gov.vn/jsp/rest.jsp"
DEFAULT_PROVIDER = "dvcquocgia"
DEFAULT_REFERER = (
    "https://vpcp.dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh.html"
)

#: Named "ref" services discovered in the portal's inline JS. Pinned as
#: constants so a portal-side version bump (``_v2`` -> ``_v3``) is a
#: one-line config/edit, not a code hunt.
SERVICE_SEARCH = "procedure_advanced_search_service_v2"
SERVICE_NEW = "procedure_get_new_procs_service_v2"
SERVICE_AGENCIES = "procedure_get_list_agency_by_type_service_v2"
SERVICE_FIELDS = "procedure_get_list_field_service_v2"
SERVICE_OBJECTS = "procedure_get_list_object_service_v2"
#: Per-procedure detail. The exact id-param is portal-version-specific
#: and is left configurable (``cfg.scraper.detail_id_param``); the
#: search service already returns the full curatable metadata, so detail
#: enrichment is opt-in (``cfg.scraper.fetch_detail``).
SERVICE_DETAIL = "proc_id_service"

#: Solr full-text source for procedures.
FTS_SOURCE = "thu_tuc_v1"


def post_params(
    session: PoliteSession, rest_url: str, params: dict[str, Any], referer: str
) -> Any:
    """POST one ``params`` object to ``rest.jsp`` and return parsed JSON."""
    resp = session.post(
        rest_url,
        data={"params": json.dumps(params, ensure_ascii=False)},
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": referer},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"rest.jsp HTTP {resp.status_code} for {params.get('service') or params.get('source_data')}")
    return resp.json()


def call_ref(
    session: PoliteSession,
    rest_url: str,
    service: str,
    *,
    referer: str = DEFAULT_REFERER,
    provider: str = DEFAULT_PROVIDER,
    **extra: Any,
) -> list[dict[str, Any]]:
    """Call a ``type="ref"`` service; always returns a list of records."""
    params: dict[str, Any] = {
        "service": service,
        "provider": provider,
        "type": "ref",
        "is_connected": 0,
        **extra,
    }
    out = post_params(session, rest_url, params, referer)
    if isinstance(out, list):
        return out
    if isinstance(out, dict) and isinstance(out.get("data"), list):
        return out["data"]
    return []


def search_page(
    session: PoliteSession,
    rest_url: str,
    *,
    page_index: int,
    record_per_page: int,
    service: str = SERVICE_SEARCH,
    keyword: str = "",
    agency_type: int = 1,
    impl_agency_id: int = -1,
    object_id: int = -1,
    field_id: int = -1,
    impl_level_id: int = -1,
    referer: str = DEFAULT_REFERER,
) -> list[dict[str, Any]]:
    """One page of the advanced-search service (the corpus spine)."""
    return call_ref(
        session,
        rest_url,
        service,
        referer=referer,
        recordPerPage=record_per_page,
        pageIndex=page_index,
        keyword=keyword,
        agency_type=agency_type,
        impl_agency_id=impl_agency_id,
        object_id=object_id,
        field_id=field_id,
        impl_level_id=impl_level_id,
    )


# ----------------------------------------------------- page-locator codec
#
# Curator's URLGenerator/DocumentDownloader contract is string-in,
# string-out. We pack the ``(agency_type, impl_agency_id, page_index)``
# coordinates of a search POST into a stable pseudo-URL so the two
# stages stay decoupled and the on-disk filename is derivable.


def encode_page_url(
    rest_url: str, *, agency_type: int, impl_agency_id: int, page_index: int
) -> str:
    q = urlencode(
        {
            "__dvc_at": agency_type,
            "__dvc_aid": impl_agency_id,
            "__dvc_page": page_index,
        }
    )
    sep = "&" if "?" in rest_url else "?"
    return f"{rest_url}{sep}{q}"


def decode_page_url(url: str) -> dict[str, int]:
    q = parse_qs(url.split("?", 1)[1]) if "?" in url else {}
    return {
        "agency_type": int(q.get("__dvc_at", ["1"])[0]),
        "impl_agency_id": int(q.get("__dvc_aid", ["-1"])[0]),
        "page_index": int(q.get("__dvc_page", ["1"])[0]),
    }


def page_doc_name(loc: dict[str, int]) -> str:
    """Stable on-disk stem for one search page (one downloaded file)."""
    return f"at{loc['agency_type']}_aid{loc['impl_agency_id']}_p{loc['page_index']:05d}"


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_REFERER",
    "DEFAULT_REST_URL",
    "FTS_SOURCE",
    "SERVICE_AGENCIES",
    "SERVICE_DETAIL",
    "SERVICE_FIELDS",
    "SERVICE_NEW",
    "SERVICE_OBJECTS",
    "SERVICE_SEARCH",
    "call_ref",
    "decode_page_url",
    "encode_page_url",
    "page_doc_name",
    "post_params",
    "search_page",
]
