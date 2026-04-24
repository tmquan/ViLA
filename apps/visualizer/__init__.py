"""Visualizer app: load the curation pipeline's parquet output and render.

Reads ``data/<host>/parquet/*.parquet`` (emitted by the pipeline's
:class:`ParquetWriter`) and runs every registered :class:`Renderer`
against it. Writes artifacts under ``data/<host>/viz/``.

CLI entry point: :mod:`apps.visualizer.__main__`.
"""
