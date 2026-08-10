"""Run several datasite crawlers concurrently, with retries + per-site logs.

This is the operational front-end for the "parallelize the HTML crawlers"
workflow: it launches ``python -m packages.datasites.<site> --pipeline
<stage>`` for each requested site as an isolated subprocess, in parallel,
and collects a structured summary (return code, wall-time, parsed scope
count, log path) for each.

Design choices, and why:

* **Subprocess isolation, not threads-in-one-process.** Each crawler
  already owns its own rate-limited session / Ray bootstrap / Playwright
  context. Running them as separate processes keeps one site's crash (or
  a leaked browser) from taking down its siblings, and lets the OS
  reclaim everything on exit. Concurrency here is process-level.
* **Scope stage by default.** Every site exposes a *cheap* first stage
  that only enumerates the corpus (``harvest`` walks the sitemap/listing,
  ``tree`` walks the codification tree) without fetching document bodies.
  That is the "dry-run scope-check": it estimates how big a full crawl
  would be and confirms the site is reachable + the pipeline is wired,
  WITHOUT launching the long ``detail`` crawl. Full crawls are opt-in via
  ``--stage detail`` (or ``all``) and a raised ``--timeout``.
* **Retries with exponential backoff**, but a *timeout* is treated as a
  partial success (the enumeration is resumable), not retried -- retrying
  a timeout just burns the target site's goodwill.

Examples::

    # Bounded parallel scope-check of all three reachable crawlers
    python scripts/crawl_parallel.py --output ~/data --timeout 180

    # Just two sites, higher per-site timeout
    python scripts/crawl_parallel.py --sites vbpl phapdien --timeout 300

    # Opt in to a bounded real detail crawl (10 docs each)
    python scripts/crawl_parallel.py --stage detail --limit 10 --timeout 600
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("crawl_parallel")

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """How to run a site's cheap scope stage + how to read its count."""

    site: str
    scope_stage: str          # the cheap enumeration stage
    count_regex: str          # pulls a corpus-size estimate from the log
    host: str                 # config host, for the reachability note
    extra_overrides: tuple[str, ...] = ()
    notes: str = ""


# The three crawlers the task parallelizes. ``count_regex`` matches the
# "stage complete" line each ``run_harvest`` / ``run_tree`` logs.
SITES: dict[str, SiteSpec] = {
    "pbgdpl": SiteSpec(
        site="pbgdpl",
        scope_stage="harvest",
        count_regex=r"harvest complete:\s*items=(\d+)",
        host="pbgdpl.gov.vn",
        # Skip the per-LinhVuc taxonomy walk during a scope probe: we only
        # want the total item count, not per-row taxonomy tags.
        extra_overrides=("scraper.walk_lv=false",),
        notes="HTML Q&A; harvest enumerates the global listing.",
    ),
    "phapdien": SiteSpec(
        site="phapdien",
        scope_stage="tree",
        # phapdien logs "tree written: <path> (245 nodes)".
        count_regex=r"\((\d+)\s*nodes\)",
        host="phapdien.moj.gov.vn",
        notes="Codification tree + articles; 'tree' walks the structure.",
    ),
    "vbpl": SiteSpec(
        site="vbpl",
        scope_stage="harvest",
        count_regex=r"harvest complete:\s*rows=(\d+)",
        host="vbpl.vn",
        notes="Sitemap harvest is cheap (~30s). NOTE: the 'detail' stage "
              "needs `playwright install chromium`; 'embed' can use the "
              "local nvidia/Nemotron-3-Embed models via embedder.runtime=hf.",
    ),
}


@dataclasses.dataclass
class SiteResult:
    site: str
    stage: str
    returncode: int | None
    status: str               # ok | failed | timeout
    seconds: float
    attempts: int
    scope_count: int | None
    log_path: str


def _parse_count(log_text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def run_one(
    spec: SiteSpec,
    *,
    stage: str,
    limit: int | None,
    output_dir: Path,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    log_dir: Path,
) -> SiteResult:
    """Run one site's crawler stage as a subprocess, with retry/backoff."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.site}.{stage}.log"

    cmd = [
        sys.executable, "-m", f"packages.datasites.{spec.site}",
        "--pipeline", stage,
        "--output", str(output_dir),
        "--log-level", "INFO",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    for ov in spec.extra_overrides:
        cmd += ["--override", ov]

    started = time.time()
    attempts = 0
    last_rc: int | None = None
    status = "failed"

    while attempts <= retries:
        attempts += 1
        logger.info("[%s] attempt %d/%d: %s", spec.site, attempts, retries + 1, " ".join(cmd))
        try:
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"# cmd: {' '.join(cmd)}\n# cwd: {REPO_ROOT}\n\n")
                fh.flush()
                proc = subprocess.run(
                    cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                    timeout=timeout_s, check=False,
                )
            last_rc = proc.returncode
            if last_rc == 0:
                status = "ok"
                break
            logger.warning("[%s] exited rc=%d (attempt %d)", spec.site, last_rc, attempts)
        except subprocess.TimeoutExpired:
            # A timeout on an enumeration stage is a *partial* result, not
            # a hard failure: the crawler caches to disk and is resumable.
            status = "timeout"
            logger.warning("[%s] timed out after %.0fs (partial, resumable)", spec.site, timeout_s)
            break

        if attempts <= retries:
            delay = backoff_s * (2 ** (attempts - 1))
            logger.info("[%s] backing off %.1fs before retry", spec.site, delay)
            time.sleep(delay)

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return SiteResult(
        site=spec.site,
        stage=stage,
        returncode=last_rc,
        status=status,
        seconds=round(time.time() - started, 1),
        attempts=attempts,
        scope_count=_parse_count(log_text, spec.count_regex),
        log_path=str(log_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES),
                        help="Which crawlers to run (default: all).")
    parser.add_argument("--stage", default="scope",
                        help="Pipeline stage per site. 'scope' (default) picks each "
                             "site's cheap enumeration stage (harvest/tree). Any other "
                             "value (detail/parse/all/...) is passed through verbatim.")
    parser.add_argument("--limit", type=int, default=None, help="Bound each crawl to N items.")
    parser.add_argument("--output", type=Path, default=Path("~/data").expanduser(),
                        help="Dataset root (default: ~/data).")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-site hard timeout (s).")
    parser.add_argument("--retries", type=int, default=1, help="Retries per site on non-zero exit.")
    parser.add_argument("--backoff", type=float, default=5.0, help="Base backoff (s), doubled per retry.")
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrency (default: one per site).")
    parser.add_argument("--log-dir", type=Path, default=None, help="Where per-site logs go.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output_dir = args.output.expanduser().resolve()
    log_dir = (args.log_dir or (output_dir / "_crawl_logs")).expanduser().resolve()
    specs = [SITES[s] for s in args.sites]
    workers = args.max_workers or len(specs)

    logger.info("output=%s  log_dir=%s  stage=%s  timeout=%ss  sites=%s",
                output_dir, log_dir, args.stage, args.timeout, args.sites)

    results: list[SiteResult] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one, spec,
                stage=(spec.scope_stage if args.stage == "scope" else args.stage),
                limit=args.limit, output_dir=output_dir, timeout_s=args.timeout,
                retries=args.retries, backoff_s=args.backoff, log_dir=log_dir,
            ): spec.site
            for spec in specs
        }
        for fut in cf.as_completed(futures):
            res = fut.result()
            results.append(res)
            logger.info("[%s] DONE status=%s rc=%s scope_count=%s (%.1fs, %d attempt(s)) -> %s",
                        res.site, res.status, res.returncode, res.scope_count,
                        res.seconds, res.attempts, res.log_path)

    results.sort(key=lambda r: r.site)
    summary = {"stage": args.stage, "output": str(output_dir), "results": [dataclasses.asdict(r) for r in results]}
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== parallel crawl summary ===")
    print(f"{'site':<10}{'stage':<9}{'status':<9}{'rc':<5}{'scope_count':<13}{'secs':<8}attempts")
    for r in results:
        print(f"{r.site:<10}{r.stage:<9}{r.status:<9}{str(r.returncode):<5}"
              f"{str(r.scope_count):<13}{r.seconds:<8}{r.attempts}")
    print(f"\nsummary  -> {summary_path}")
    print(f"per-site -> {log_dir}/<site>.<stage>.log")

    # Exit non-zero only if a site hard-failed (timeouts are acceptable
    # for a bounded scope probe).
    return 0 if all(r.status in ("ok", "timeout") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
