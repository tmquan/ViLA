"""Consumer apps that read the Curator pipeline's final parquet output.

These live outside the pipeline (no ``ProcessingStage`` subclasses) so
they can be iterated independently of the data-curation cadence.
"""
