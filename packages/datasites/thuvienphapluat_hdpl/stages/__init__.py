"""hoi-dap Q&A Curator ``ProcessingStage``s (sibling of ``components/``).

``components/`` holds the four Curator primitives (URLGenerator / Downloader /
Iterator / Extractor); this package holds the stages that compose or wrap them:

    extractor.py  -- TVPLQAExtractStage  (pages/*.html.gz -> Q&A records)
    embedder.py   -- TVPLQAEmbedStage    (Q&A -> question/answer embeddings)
"""
from packages.datasites.thuvienphapluat_hdpl.stages.embedder import TVPLQAEmbedStage
from packages.datasites.thuvienphapluat_hdpl.stages.extractor import TVPLQAExtractStage

__all__ = ["TVPLQAEmbedStage", "TVPLQAExtractStage"]
