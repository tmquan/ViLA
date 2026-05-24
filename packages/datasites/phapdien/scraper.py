"""Crawler for the public Bộ pháp điển corpus on phapdien.moj.gov.vn.

The site is an ASP.NET WebForms application. The useful legal corpus is
not rendered directly on the landing page; the browser first loads the
tree from ``TreeBoPD.aspx``, then opens each ``Đề mục`` through
``ViewBoPD.aspx``. That view page contains a ``fileVersion`` value used
by ``ActionHandler.aspx`` to return the full codified legal text as HTML.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from bs4 import BeautifulSoup, Tag

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.phapdien._shared import (
    ARTICLE_FIELDS,
    SUBJECT_FIELDS,
    TREE_NODE_FIELDS,
    build_layout,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://phapdien.moj.gov.vn"
TREE_URL = f"{BASE_URL}/TraCuuPhapDien/TreeBoPD.aspx"
VIEW_URL = f"{BASE_URL}/TraCuuPhapDien/ViewBoPD.aspx"
ACTION_URL = f"{BASE_URL}/TraCuuPhapDien/ActionHandler.aspx"

_TREE_JSON_RE = re.compile(r"CreateTree',\s*'(.+?)'\s*,", re.S)
_FILE_VERSION_RE = re.compile(r"fileVersion:\s*'([^']+)'")
_WORD_RE = re.compile(r"\S+", re.UNICODE)


@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    kind: str
    number: str
    title: str
    raw_text: str


@dataclass
class SubjectNode:
    subject_id: str
    topic_id: str | None
    topic_number: str
    topic_title: str
    subject_number: str
    subject_title: str


class PhapdienCrawler:
    """Harvest the tree, full đề mục HTML, and article-level JSONL."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._cache_html = bool(cfg.scraper.get("cache_details", True))
        self._limit = cfg.get("limit", None)
        self._session: PoliteSession | None = None

    def run_tree(self) -> Path:
        """Fetch and parse the topic/de-muc tree."""
        session = self._ensure_session()
        tree_html = self._fetch_tree_html(session)
        nodes = parse_tree_nodes(tree_html)
        out = self.layout.jsonl_dir / "tree_nodes.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for node in nodes:
                f.write(
                    json.dumps(
                        {k: getattr(node, k) for k in TREE_NODE_FIELDS},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("tree written: %s (%d nodes)", out, len(nodes))
        return out

    def run_detail(self) -> Path:
        """Fetch full HTML for each de-muc and write de-muc/articles JSONL."""
        session = self._ensure_session()
        tree_html = self._fetch_tree_html(session)
        nodes = parse_tree_nodes(tree_html)
        subjects = build_subject_index(nodes)
        if self._limit is not None:
            subjects = subjects[: int(self._limit)]

        subject_out = self.layout.jsonl_dir / "subjects.jsonl"
        article_out = self.layout.jsonl_dir / "articles.jsonl"
        ok = err = article_count = 0
        scraped_at = _utc_now_iso()

        with subject_out.open("w", encoding="utf-8") as subject_f, article_out.open(
            "w",
            encoding="utf-8",
        ) as article_f:
            for idx, subject in enumerate(subjects, 1):
                meta, articles = self._fetch_one_subject(session, subject, scraped_at)
                subject_f.write(
                    json.dumps(
                        {k: meta.get(k) for k in SUBJECT_FIELDS},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for article in articles:
                    article_f.write(
                        json.dumps(
                            {k: article.get(k) for k in ARTICLE_FIELDS},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if meta["fetch_status"] == "ok":
                    ok += 1
                    article_count += len(articles)
                else:
                    err += 1
                if idx % 10 == 0:
                    logger.info(
                        "detail progress: %d/%d subjects ok=%d err=%d articles=%d",
                        idx, len(subjects), ok, err, article_count,
                    )

        self._write_manifest(
            subjects_total=len(subjects),
            subjects_ok=ok,
            subjects_err=err,
            articles_total=article_count,
        )
        logger.info(
            "detail written: %s, %s (subjects=%d ok=%d err=%d articles=%d)",
            subject_out, article_out, len(subjects), ok, err, article_count,
        )
        return article_out

    def _fetch_one_subject(
        self,
        session: PoliteSession,
        subject: SubjectNode,
        scraped_at: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        view_url = f"{VIEW_URL}?{urlencode({'obj': '', 'demucid': subject.subject_id, 'mapc': '1'})}"
        view_path = self.layout.html_dir / "view" / f"{subject.subject_id}.html"
        content_path = self.layout.html_dir / "content" / f"{subject.subject_id}.html"
        md_path = self.layout.md_dir / f"{subject.subject_id}.md"
        base_meta = {
            "subject_id": subject.subject_id,
            "topic_id": subject.topic_id,
            "topic_number": subject.topic_number,
            "topic_title": subject.topic_title,
            "subject_number": subject.subject_number,
            "subject_title": subject.subject_title,
            "source_url": view_url,
            "view_html_path": str(view_path.resolve()),
            "content_html_path": str(content_path.resolve()),
            "markdown_path": str(md_path.resolve()),
            "file_version": None,
            "fetch_status": "ok",
            "fetch_error": None,
            "scraped_at": scraped_at,
        }
        try:
            view_html = self._get_or_post(view_path, session, view_url, data={})
            m = _FILE_VERSION_RE.search(view_html)
            if not m:
                raise RuntimeError("fileVersion not found in ViewBoPD.aspx")
            file_version = m.group(1)
            base_meta["file_version"] = file_version
            if content_path.exists() and content_path.stat().st_size > 0:
                content_html = content_path.read_text(encoding="utf-8")
            else:
                resp = session.post(
                    ACTION_URL,
                    data={
                        "deMucID": subject.subject_id,
                        "fileVersion": file_version,
                        "do": "html",
                    },
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"ActionHandler.aspx HTTP {resp.status_code}")
                payload = resp.json()
                if payload.get("Erros"):
                    raise RuntimeError(payload.get("Message") or "ActionHandler error")
                content_html = payload.get("Data") or ""
                if not content_html:
                    raise RuntimeError("ActionHandler returned empty Data")
                if self._cache_html:
                    content_path.write_text(content_html, encoding="utf-8")
            md_path.write_text(html_to_markdown_text(content_html), encoding="utf-8")
            articles = parse_articles(content_html, subject, view_url, scraped_at)
            return base_meta, articles
        except Exception as exc:
            logger.exception("subject failed: %s", subject.subject_id)
            base_meta["fetch_status"] = f"error:{type(exc).__name__}"
            base_meta["fetch_error"] = str(exc)
            return base_meta, []

    def _fetch_tree_html(self, session: PoliteSession) -> str:
        path = self.layout.html_dir / "tree.html"
        if path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
        resp = session.post(TREE_URL, data={})
        if resp.status_code != 200:
            raise RuntimeError(f"TreeBoPD.aspx HTTP {resp.status_code}")
        path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def _get_or_post(
        self,
        path: Path,
        session: PoliteSession,
        url: str,
        *,
        data: dict[str, str],
    ) -> str:
        if self._cache_html and path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
        resp = session.post(url, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"{url} HTTP {resp.status_code}")
        if self._cache_html:
            path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def _write_manifest(
        self,
        *,
        subjects_total: int,
        subjects_ok: int,
        subjects_err: int,
        articles_total: int,
    ) -> None:
        payload = {
            "host": self.layout.host,
            "completed_at": _utc_now_iso(),
            "subjects_total": subjects_total,
            "subjects_ok": subjects_ok,
            "subjects_err": subjects_err,
            "articles_total": articles_total,
            "tree_nodes_jsonl": str((self.layout.jsonl_dir / "tree_nodes.jsonl").resolve()),
            "subjects_jsonl": str((self.layout.jsonl_dir / "subjects.jsonl").resolve()),
            "articles_jsonl": str((self.layout.jsonl_dir / "articles.jsonl").resolve()),
        }
        (self.layout.jsonl_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_session(self) -> PoliteSession:
        if self._session is None:
            self._session = session_from_scraper_cfg(self.cfg)
        return self._session


def parse_tree_nodes(tree_html: str) -> list[TreeNode]:
    """Extract the initial jstree payload embedded in ``TreeBoPD.aspx``."""
    match = _TREE_JSON_RE.search(tree_html)
    if not match:
        raise ValueError("Could not locate CreateTree JSON in TreeBoPD.aspx")
    raw_json = match.group(1).replace("\\'", "'")
    payload = json.loads(raw_json)
    nodes: list[TreeNode] = []
    for item in payload:
        cls = (item.get("li_attr") or {}).get("class") or ""
        kind = "topic" if cls == "treenode-chude" else "subject"
        raw_text = _text_without_action_links(item.get("text") or "")
        number, title = _split_numbered_title(raw_text, kind)
        parent = item.get("parent")
        nodes.append(
            TreeNode(
                node_id=str(item["id"]),
                parent_id=None if parent == "#" else str(parent),
                kind=kind,
                number=number,
                title=title,
                raw_text=raw_text,
            )
        )
    return nodes


def build_subject_index(nodes: list[TreeNode]) -> list[SubjectNode]:
    topics = {n.node_id: n for n in nodes if n.kind == "topic"}
    out: list[SubjectNode] = []
    for node in nodes:
        if node.kind != "subject":
            continue
        topic = topics.get(node.parent_id or "")
        out.append(
            SubjectNode(
                subject_id=node.node_id,
                topic_id=node.parent_id,
                topic_number=topic.number if topic else "",
                topic_title=topic.title if topic else "",
                subject_number=node.number,
                subject_title=node.title,
            )
        )
    return sorted(out, key=lambda d: (int(d.topic_number or 0), _sortable_number(d.subject_number), d.subject_title))


def parse_articles(
    content_html: str,
    subject: SubjectNode,
    source_url: str,
    scraped_at: str,
) -> list[dict[str, Any]]:
    """Split one de-muc HTML document into article-level records."""
    soup = BeautifulSoup(content_html, "html.parser")
    records: list[dict[str, Any]] = []
    chapter_parts: list[str] = []
    for p in soup.find_all("p"):
        classes = set(p.get("class") or [])
        if "pChuong" in classes:
            text = _clean_text(p.get_text(" "))
            if text:
                if text.lower().startswith("chương"):
                    chapter_parts = [text]
                elif chapter_parts:
                    chapter_parts.append(text)
            continue
        if "pDieu" not in classes:
            continue
        article_title = _clean_text(p.get_text(" "))
        anchor = ""
        a = p.find("a", attrs={"name": True})
        if a is not None:
            anchor = str(a.get("name") or "")

        source_note = ""
        related_note = ""
        content_chunks: list[str] = []
        source_links: list[dict[str, str]] = []
        for sib in p.next_siblings:
            if isinstance(sib, Tag):
                sib_classes = set(sib.get("class") or [])
                if "pDieu" in sib_classes or "pChuong" in sib_classes:
                    break
                text = _clean_text(sib.get_text(" "))
                if not text:
                    continue
                if "pGhiChu" in sib_classes:
                    source_note = text
                    source_links = [
                        {"text": _clean_text(a.get_text(" ")), "href": str(a.get("href") or "")}
                        for a in sib.find_all("a")
                    ]
                elif "pChiDan" in sib_classes:
                    related_note = text
                else:
                    content_chunks.append(text)
        content_text = "\n".join(content_chunks).strip()
        records.append(
            {
                "subject_id": subject.subject_id,
                "topic_id": subject.topic_id,
                "topic_number": subject.topic_number,
                "topic_title": subject.topic_title,
                "subject_number": subject.subject_number,
                "subject_title": subject.subject_title,
                "article_anchor": anchor,
                "article_title": article_title,
                "chapter_title": " - ".join(chapter_parts),
                "source_note_text": source_note,
                "source_links": source_links,
                "related_note_text": related_note,
                "content_text": content_text,
                "content_char_len": len(content_text),
                "content_word_count": len(_WORD_RE.findall(content_text)),
                "source_url": f"{source_url}#{anchor}" if anchor else source_url,
                "scraped_at": scraped_at,
            }
        )
    return records


def html_to_markdown_text(content_html: str) -> str:
    soup = BeautifulSoup(content_html, "html.parser")
    for tag in soup(["script", "style", "link"]):
        tag.decompose()
    lines: list[str] = []
    for elem in soup.find_all(["h1", "h2", "h3", "p"]):
        text = _clean_text(elem.get_text(" "))
        if not text:
            continue
        classes = set(elem.get("class") or [])
        if elem.name in {"h1", "h2", "h3"}:
            lines.extend(["", f"# {text}", ""])
        elif "pChuong" in classes or "pDieu" in classes:
            lines.extend(["", f"## {text}", ""])
        else:
            lines.append(text)
    return "\n".join(lines).strip() + "\n"


def run_tree(cfg: Any) -> Path:
    return PhapdienCrawler(cfg, build_layout(cfg)).run_tree()


def run_detail(cfg: Any) -> Path:
    return PhapdienCrawler(cfg, build_layout(cfg)).run_detail()


PIPELINES = {
    "tree": run_tree,
    "detail": run_detail,
}
ALL_PIPELINES_ORDER = ["tree", "detail"]


def run_pipeline(cfg: Any, name: str) -> Path:
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline {name!r}; choices: {list(PIPELINES) + ['all']}")
    return PIPELINES[name](cfg)


def _text_without_action_links(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for a in soup.find_all("a"):
        a.decompose()
    return _clean_text(soup.get_text(" "))


def _split_numbered_title(text: str, kind: str) -> tuple[str, str]:
    label = "Chủ đề" if kind == "topic" else "Đề mục"
    match = re.match(rf"{label}\s+số\s+([^:]+):\s*(.+)$", text, re.I)
    if not match:
        return "", text
    return match.group(1).strip(), match.group(2).strip()


def _sortable_number(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
        return 10_000, digest


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "PhapdienCrawler",
    "build_subject_index",
    "parse_articles",
    "parse_tree_nodes",
    "run_detail",
    "run_pipeline",
    "run_tree",
]

