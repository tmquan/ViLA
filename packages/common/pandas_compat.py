"""pandas 3.0 compatibility shim for the ViLA pipeline.

pandas 3.0 promotes **PyArrow-backed strings to the default**: an
object-dtype string column is inferred as Arrow ``large_string`` at
``DataFrame`` *construction* time, and ``pyarrow.array`` rejects lone
UTF-16 surrogates (the U+D800..U+DFFF range) with ``UnicodeEncodeError``.

That breaks this pipeline because upstream Vietnamese-legal PDFs
occasionally carry a malformed UTF-16 ``/Info`` field where one half of
a surrogate pair was dropped in transit (~1 row per ~1M docs on the live
congbobanan parse). Under pandas 2.x those surrogates rode along in an
object column until :func:`packages.pipeline.io._scrub_surrogates`
replaced them with U+FFFD at write time. Under pandas 3.0 the frame can
no longer even be *built*, so the crash moves upstream of every scrub
the pipeline already performs -- and the two ``test_markdown_io``
surrogate regression tests fail at their own ``pd.DataFrame(...)`` setup.

The entire codebase (and its 400+ tests) was written against pandas 2.x
object-string semantics. Rather than sprinkle Arrow-safe construction
across every stage, we restore the historical behaviour process-wide by
opting out of the new string inference. This keeps the surrogate defense
where it already lives (the writers) and touches no stage logic.

Called once as an import side effect of :mod:`packages.common`, which
every datasite, stage, and the test ``conftest`` import -- so it applies
uniformly in the CLI, in Ray/xenna workers (which re-import the stage
modules), and under pytest.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure_pandas() -> None:
    """Opt out of pandas 3.0 Arrow-backed string inference (idempotent).

    No-op on pandas < 3.0 (the option does not exist) and safe to call
    repeatedly. We swallow the lookup error rather than pin a pandas
    version so the shim is forward/backward compatible.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dep
        return
    try:
        pd.set_option("future.infer_string", False)
    except (KeyError, ValueError):  # pragma: no cover - option absent on this pandas
        logger.debug("pandas has no 'future.infer_string' option; skipping shim")


__all__ = ["configure_pandas"]
