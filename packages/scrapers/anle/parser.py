"""Parser for anle PDFs via nvidia/nemotron-parse NIM.

Stage 2 of the anle pipeline. Reads downloaded PDFs under
data/anle.toaan.gov.vn/pdf/ and writes per-document structured layout
and markdown outputs:

    data/anle.toaan.gov.vn/
        md/<doc_id>.md          # full markdown body (all pages joined)
        json/<doc_id>.json      # structured layout (blocks + bbox per page)
        progress.parse.json
        logs/parse-<date>.jsonl

The NIM client is abstracted behind `NemotronParseClient` so unit tests
can inject a fake. See tests/unit/scrapers/anle/test_parser.py.

Run:
    export NVIDIA_API_KEY=nvapi-...
    python -m packages.scrapers.anle.parser --config-name anle --num-workers 4
    python -m packages.scrapers.anle.parser --limit 5
    python -m packages.scrapers.anle.parser --doc TAND349038
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.common.stages import StageBase

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


# ----------------------------------------------------------------- client


class NemotronParseClient(Protocol):
    """Minimal surface expected from a nemotron-parse client."""

    def parse(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        """Return a dict with keys: pages, markdown, confidence."""


class NimNemotronParseClient:
    """Thin wrapper over the NIM nemotron-parse endpoint.

    Uses direct `requests.post` rather than the OpenAI SDK because
    nemotron-parse is a CV/document endpoint under
    `ai.api.nvidia.com/v1/cv/`, not a chat/completions model. The
    response shape varies by NIM release; we normalize to
    {pages: [...], markdown: str, confidence: float | None}.
    """

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
        self._model = model
        self._timeout = timeout

    def parse(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        import base64

        payload_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        url = f"{self._base_url}/cv/{self._model}"
        resp = self._session.post(
            url,
            json={
                "input": [{"type": "file", "data": payload_b64, "mime_type": "application/pdf"}],
                "options": {"preserve_tables": preserve_tables, "emit_layout": True},
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return _normalize_nemotron_response(resp.json())


class LocalPdfParseClient:
    """Pure-Python local parser supporting PDF / DOCX / DOC.

    anle's Oracle WebCenter backend serves most precedents as PDF but a
    few drafts / foreign precedents as DOCX (or legacy DOC). The parse
    interface takes raw bytes and dispatches by magic-number:

        %PDF       -> pypdf
        PK\\x03     -> docx2txt (DOCX is a ZIP)
        else       -> best-effort (log warning, return empty)
    """

    def __init__(self) -> None:
        try:
            import pypdf  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "runtime=local requires `pypdf`. Install with `pip install pypdf`."
            ) from exc

    def parse(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        head = pdf_bytes[:4]
        if head.startswith(b"%PDF"):
            return self._parse_pdf(pdf_bytes)
        if head.startswith(b"PK\x03\x04"):
            return self._parse_docx(pdf_bytes)
        logger.warning(
            "LocalPdfParseClient: unrecognized magic %r (%d bytes) — skipping",
            head, len(pdf_bytes),
        )
        return {"pages": [], "markdown": "", "confidence": None}

    @staticmethod
    def _parse_pdf(data: bytes) -> dict[str, Any]:
        import io

        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages: list[dict[str, Any]] = []
        md_parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            md = text.strip()
            pages.append({"page_number": i, "markdown": md, "blocks": []})
            if md:
                md_parts.append(f"## Page {i}\n\n{md}")
        return {"pages": pages, "markdown": "\n\n".join(md_parts), "confidence": None}

    @staticmethod
    def _parse_docx(data: bytes) -> dict[str, Any]:
        import io

        try:
            import docx2txt
        except ImportError:  # pragma: no cover
            logger.warning("docx2txt not installed — skipping DOCX")
            return {"pages": [], "markdown": "", "confidence": None}
        text = docx2txt.process(io.BytesIO(data)) or ""
        text = text.strip()
        # DOCX has no native paging; treat the whole document as one logical page.
        if not text:
            return {"pages": [], "markdown": "", "confidence": None}
        return {
            "pages": [{"page_number": 1, "markdown": text, "blocks": []}],
            "markdown": f"## Page 1\n\n{text}",
            "confidence": None,
        }


def _normalize_nemotron_response(resp: Any) -> dict[str, Any]:
    """Coerce a nemotron-parse response into the shape ViLA expects."""
    if isinstance(resp, dict):
        return resp
    # SDK may return a typed object; best-effort conversion.
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))


# ----------------------------------------------------------------- parser


@dataclass
class ParseResult:
    doc_id: str
    markdown: str
    pages: list[dict[str, Any]]
    num_pages: int
    confidence: float | None
    parser_model: str
    parsed_at: str
    source_filename: str


class AnleParser(StageBase):
    """Runs nemotron-parse across data/<host>/pdf/ and writes md/ + json/."""

    stage = "parse"
    required_dirs = ("md_dir", "json_dir", "metadata_dir", "logs_dir")
    uses_progress = True

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        client: NemotronParseClient,
        *,
        num_workers: int = 4,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
        doc_filter: str | None = None,
    ) -> None:
        super().__init__(cfg, layout, force=force, resume=resume, limit=limit)
        self.client = client
        self.num_workers = num_workers
        self.doc_filter = doc_filter

    def is_item_complete(self, doc_id: str) -> bool:
        md = self.layout.md_dir / f"{doc_id}.md"
        js = self.layout.json_dir / f"{doc_id}.json"
        if not (md.exists() and js.exists()):
            return False
        if md.stat().st_size == 0 or js.stat().st_size == 0:
            return False
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
            return int(data.get("num_pages", 0)) > 0
        except (json.JSONDecodeError, OSError):
            return False

    def iter_pdfs(self) -> list[Path]:
        # anle serves PDF, DOCX, and a few legacy DOC; LocalPdfParseClient
        # dispatches by magic-number.
        files: list[Path] = []
        for ext in ("*.pdf", "*.docx", "*.doc"):
            files.extend(self.layout.pdf_dir.glob(ext))
        files = sorted(set(files))
        if self.doc_filter:
            files = [p for p in files if p.stem == self.doc_filter]
        if self.limit is not None:
            files = files[: self.limit]
        return files

    def run(self) -> dict[str, int]:
        counts = {"seen": 0, "skipped": 0, "processed": 0, "errored": 0}
        pdfs = self.iter_pdfs()
        futures: list[cf.Future[ParseResult]] = []
        with cf.ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for pdf in pdfs:
                counts["seen"] += 1
                doc_id = pdf.stem
                if not self.force and self.progress.is_complete(doc_id) and self.is_item_complete(doc_id):
                    counts["skipped"] += 1
                    continue
                futures.append(ex.submit(self._parse_one, pdf))
            for fut in cf.as_completed(futures):
                try:
                    result = fut.result()
                except Exception as exc:
                    counts["errored"] += 1
                    logger.exception("parse failed")
                    self.log.error(error=str(exc))
                else:
                    self._persist(result)
                    self.progress.mark_complete(result.doc_id)
                    counts["processed"] += 1
                    self.log.info(item_id=result.doc_id, pages=result.num_pages)
        self.log.info(event="run_done", **counts)
        return counts

    def _parse_one(self, pdf: Path) -> ParseResult:
        pdf_bytes = pdf.read_bytes()
        resp = self.client.parse(pdf_bytes, preserve_tables=True)

        pages = resp.get("pages") or []
        markdown = resp.get("markdown") or _join_markdown(pages)
        confidence = resp.get("confidence")

        return ParseResult(
            doc_id=pdf.stem,
            markdown=markdown,
            pages=pages,
            num_pages=len(pages) if pages else _count_markdown_pages(markdown),
            confidence=float(confidence) if confidence is not None else None,
            parser_model=str(self.cfg.parser.model_id),
            parsed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_filename=pdf.name,
        )

    def _persist(self, r: ParseResult) -> None:
        md_path = self.layout.md_dir / f"{r.doc_id}.md"
        md_path.write_text(r.markdown, encoding="utf-8")

        meta_path = self.layout.metadata_dir / f"{r.doc_id}.json"
        scraper_metadata: dict[str, Any] = {}
        if meta_path.exists():
            try:
                scraper_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                scraper_metadata = {}

        payload = {
            "doc_id": r.doc_id,
            "source_pdf": str((self.layout.pdf_dir / r.source_filename).relative_to(
                self.layout.output_root
            )),
            "model": r.parser_model,
            "parsed_at": r.parsed_at,
            "confidence": r.confidence,
            "num_pages": r.num_pages,
            "metadata": scraper_metadata,
            "pages": r.pages,
            "markdown": r.markdown,
        }
        json_path = self.layout.json_dir / f"{r.doc_id}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _join_markdown(pages: list[dict[str, Any]]) -> str:
    parts = []
    for p in pages:
        md = p.get("markdown") or ""
        if md:
            parts.append(md.strip())
    return "\n\n".join(parts)


def _count_markdown_pages(markdown: str) -> int:
    if not markdown:
        return 0
    # Page boundary heuristic: form-feed or "--- page N ---" markers.
    return max(1, markdown.count("\f") + 1)


# ----------------------------------------------------------------- CLI


def _build_client(cfg: Any) -> NemotronParseClient:
    runtime = str(cfg.parser.runtime).lower()
    if runtime == "local":
        return LocalPdfParseClient()
    if runtime == "nim":
        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY) is required for "
                "runtime=nim. Export it or switch to parser.runtime=local."
            )
        base_url = str(cfg.parser.nim_base_url)
        if base_url.startswith("${") and base_url.endswith("}"):
            base_url = "https://ai.api.nvidia.com/v1"
        return NimNemotronParseClient(
            api_key=api_key,
            base_url=base_url,
            model=str(cfg.parser.model_id),
            timeout=float(cfg.parser.timeout_s),
        )
    raise ValueError(f"unknown parser runtime: {runtime}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Parser for anle PDFs via nvidia/nemotron-parse (stage 2).",
        stage="parse",
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Process only this doc_id (useful for debugging a single PDF).",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    cfg = load_and_override(
        config_path=config_path,
        overrides=args.override,
        schema_cls=PipelineCfg,
    )

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    client = _build_client(cfg)
    anle_parser = AnleParser(
        cfg=cfg,
        layout=layout,
        client=client,
        num_workers=int(args.num_workers or cfg.parser.num_workers),
        limit=args.limit,
        force=args.force,
        resume=not args.no_resume,
        doc_filter=args.doc,
    )
    counts = anle_parser.run()
    logger.info("parse done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
