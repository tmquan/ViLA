"""luutru.gov.vn datasite -- Vietnamese legal-document (văn bản) corpus.

State Records and Archives Department of Vietnam (Cục Văn thư và Lưu
trữ nhà nước) publishes its normative documents (văn bản QPPL) and
administrative directives (văn bản CĐĐH) as PDFs with rich metadata
behind a GET-paginated ASP.NET document-search surface
(``/vanban.aspx``). This is a **Family A** Curator datasite: the
corpus ships as PDFs, so it runs the full five-stage chain.

Top-level files map 1-to-1 onto the five Curator pipelines + the HF
publish path:

    download.py    -> URLs     -> PDFs
    parse.py       -> PDFs     -> markdown
    extract.py     -> markdown -> JSONL
    embed.py       -> JSONL    -> embeddings parquet
    reduce.py      -> embeddings parquet -> reduced parquet
    pipeline.py    -> registry + ``build_pipeline(cfg, name)`` dispatch
    hf_export.py   -> JSONL    -> hf/ (parquet + README + manifest)
    push_to_hf.py  -> hf/      -> HuggingFace dataset repo

The four Curator abstract-base subclasses (URLGenerator,
DocumentDownloader, DocumentIterator, DocumentExtractor) live under
:mod:`packages.datasites.luutru.components`.
"""

from packages.datasites.luutru.components import (
    LuutruDocumentDownloader,
    LuutruDocumentExtractor,
    LuutruDocumentIterator,
    LuutruURLGenerator,
)
from packages.datasites.luutru.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_download_pipeline,
    build_embed_pipeline,
    build_extract_pipeline,
    build_parse_pipeline,
    build_pipeline,
    build_reduce_pipeline,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "LuutruDocumentDownloader",
    "LuutruDocumentExtractor",
    "LuutruDocumentIterator",
    "LuutruURLGenerator",
    "build_download_pipeline",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_parse_pipeline",
    "build_pipeline",
    "build_reduce_pipeline",
]
