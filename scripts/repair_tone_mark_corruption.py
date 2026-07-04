"""Repair tone-mark corruption introduced by the pre-fix normalizer.

Before the closed-syllable guard landed in
``packages/extractor/normalization._build_tone_re``, the old→new
tone-mark rewrite fired inside CLOSED syllables and triphthongs,
turning correct spellings into sequences that are invalid under both
the pre- and post-1984 conventions::

    hoạt → họat   toàn → tòan   ngoài → ngòai   loại → lọai

Because a head-vowel tone mark followed by another LETTER inside the
same word is never valid Vietnamese orthography, the inverse rewrite
is unambiguous: every ``(new-form digraph)(letter)`` occurrence is
mapped back to the ``(old-form digraph)(letter)`` spelling — which for
closed syllables is the one correct form. The swap transposes two
code points and never changes string length, so ``char_len`` columns
and entity ``start`` / ``end`` offsets remain valid. Hash columns
(``text_hash``, ``embedding_text_hash``) are deliberately left
untouched: they are cross-stage join keys computed over the pre-repair
text.

Usage::

    python scripts/repair_tone_mark_corruption.py --outdir DIR FILE_OR_GLOB [...]

NON-DESTRUCTIVE: originals are never modified. A repaired copy of
each file that needed at least one replacement is written to
``--outdir`` under its original basename; files with zero hits are
skipped (nothing to swap). Review the copies, then swap them over the
originals (e.g. ``mv DIR/*.parquet <original dir>/``). A per-file
summary is printed to stdout.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from packages.extractor.normalization import _TONE_TABLE

# Inverse of the tone table: new-form digraph → old-form digraph,
# applied ONLY when a letter follows (the invalid-context signature).
_INVERSE = {new: old for old, new in _TONE_TABLE.items()}
_LETTER = r"[^\W\d_]"  # a Unicode letter (no digits / underscore)
_FIX_RE = re.compile(
    "(?:"
    + "|".join(re.escape(k) for k in sorted(_INVERSE, key=len, reverse=True))
    + f")(?={_LETTER})"
)
# RE2-compatible detection pattern for fast Arrow-side column scans.
_DETECT = (
    "(?:"
    + "|".join(re.escape(k) for k in _INVERSE)
    + r")\p{L}"
)


def fix_str(s: str) -> str:
    return _FIX_RE.sub(lambda m: _INVERSE[m.group(0)], s)


def _fix_nested(value):  # noqa: ANN001 - recursive pylist values
    if isinstance(value, str):
        return fix_str(value)
    if isinstance(value, list):
        return [_fix_nested(v) for v in value]
    if isinstance(value, dict):
        return {k: _fix_nested(v) for k, v in value.items()}
    return value


_SKIP_COLUMNS = frozenset({"text_hash", "embedding_text_hash"})


def repair_file(path: str, outdir: str) -> int:
    """Write a repaired copy of ``path`` into ``outdir`` when needed.

    Returns the number of changed cells (0 → no copy written).
    """
    table = pq.read_table(path)
    changed_cells = 0
    new_columns: list[pa.ChunkedArray] = []
    for name in table.column_names:
        col = table.column(name)
        if name in _SKIP_COLUMNS or not (
            pa.types.is_string(col.type)
            or pa.types.is_large_string(col.type)
            or pa.types.is_nested(col.type)
        ):
            new_columns.append(col)
            continue
        if pa.types.is_nested(col.type):
            # Nested structs/lists may hold text spans (e.g. NER
            # entities). Small tables only — Python round-trip.
            pylist = col.to_pylist()
            fixed = [_fix_nested(v) for v in pylist]
            n = sum(1 for a, b in zip(pylist, fixed) if a != b)
            if n:
                changed_cells += n
                new_columns.append(
                    pa.chunked_array([pa.array(fixed, type=col.type)])
                )
            else:
                new_columns.append(col)
            continue
        mask = pc.fill_null(pc.match_substring_regex(col, _DETECT), False)
        n_hit = pc.sum(mask).as_py() or 0
        if not n_hit:
            new_columns.append(col)
            continue
        # Only round-trip the affected rows through Python.
        values = col.to_pandas()
        hits = mask.to_pandas()
        values.loc[hits] = values.loc[hits].map(fix_str)
        changed_cells += int(n_hit)
        new_columns.append(
            pa.chunked_array([pa.Array.from_pandas(values, type=col.type)])
        )
    if not changed_cells:
        return 0
    repaired = pa.table(new_columns, schema=table.schema)
    out_path = os.path.join(outdir, os.path.basename(path))
    tmp = out_path + ".repair.tmp"
    pq.write_table(repaired, tmp, compression="zstd")
    os.replace(tmp, out_path)
    return changed_cells


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "--outdir":
        print(__doc__, file=sys.stderr)
        return 2
    outdir = argv[1]
    os.makedirs(outdir, exist_ok=True)
    paths: list[str] = []
    for pattern in argv[2:]:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: no files match {pattern!r}", file=sys.stderr)
        paths.extend(matches)
    total = 0
    n_written = 0
    for i, path in enumerate(paths, 1):
        n = repair_file(path, outdir)
        total += n
        n_written += bool(n)
        print(f"[{i}/{len(paths)}] {os.path.basename(path)}: "
              f"{n} cells repaired", flush=True)
    print(f"TOTAL: {total} cells repaired; {n_written}/{len(paths)} "
          f"repaired copies written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
