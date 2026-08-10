"""Mint a fresh thuvienphapluat.vn ``cf_clearance`` on the DGX itself.

The site's "Just a moment…" wall is a Cloudflare *managed challenge* (a JS
integrity check, not an interactive CAPTCHA), so a real, stealth-configured
Chromium can solve it headlessly. Driving that browser here — on the box that
will do the crawling — means the resulting ``cf_clearance`` is bound to the
DGX's own IP + TLS, and can be re-minted automatically when it ages out. No
Mac, no SSH SOCKS tunnel, no hand-copied cURL.

Reuses the stealth launch kit already proven on vbpl.vn
(:mod:`packages.datasites.vbpl.components.detail`).

    python -m packages.datasites.thuvienphapluat_vanban.mint_cookie \
        --out-cookie ~/.tvpl_cookie --out-ua ~/.tvpl_ua

Writes the full cookie header (incl. ``cf_clearance``) + the exact User-Agent
the browser presented, then prints ``MINTED`` / ``FAILED``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from packages.datasites.vbpl.components.detail import (
    _DEFAULT_CHROMIUM_ARGS,
    _STEALTH_INIT_SCRIPT,
    _autodetect_full_chromium,
)

logger = logging.getLogger(__name__)


def _async_playwright():
    """Prefer patchright (stealth Playwright drop-in) — it patches the CDP /
    ``Runtime.enable`` leaks Cloudflare fingerprints. Fall back to vanilla."""
    try:
        from patchright.async_api import async_playwright  # type: ignore
        return async_playwright, True
    except Exception:  # noqa: BLE001
        from playwright.async_api import async_playwright
        return async_playwright, False

# A page that reliably triggers the challenge (the homepage does not).
DEFAULT_CHALLENGE_URL = "https://thuvienphapluat.vn/hoi-dap-phap-luat/doanh-nghiep"

# The bundled Chromium advertises "HeadlessChrome/…", which Cloudflare blocks
# on sight. Present a real desktop-Chrome UA instead (matches chromium-1228 =
# Chrome 149). Whatever UA we mint under is written to ~/.tvpl_ua so the
# curl_cffi crawler sends the identical UA the cf_clearance is bound to.
REAL_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


def _is_challenge(title: str, content_len: int) -> bool:
    return ("just a moment" in (title or "").lower()) or content_len < 20_000


async def _mint(url: str, *, headless: bool, wait_s: int) -> tuple[list[dict], str | None, bool]:
    async_playwright, is_patchright = _async_playwright()
    logger.info("engine=%s headless=%s", "patchright" if is_patchright else "playwright", headless)

    # patchright patches stealth natively and dislikes custom args / init
    # scripts; vanilla playwright needs the args + init script.
    if is_patchright:
        args: list[str] = []
    else:
        args = list(_DEFAULT_CHROMIUM_ARGS)
        if not headless:
            args = [a for a in args if a != "--headless=new"]
    launch_kwargs: dict = {"headless": headless, "args": args}
    if is_patchright:
        # Let patchright pick its own patched Chromium (its stealth patches
        # are paired with that build). `patchright install chromium` provides it.
        logger.info("using patchright's bundled Chromium")
    else:
        exe = _autodetect_full_chromium()
        if exe:
            launch_kwargs["executable_path"] = exe
            logger.info("using full Chromium: %s", exe)
        else:
            logger.warning("full Chromium not found; headless-shell will likely be flagged")

    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        try:
            ctx = await browser.new_context(
                user_agent=REAL_UA,
                locale="vi-VN",
                timezone_id="Asia/Ho_Chi_Minh",
                viewport={"width": 1366, "height": 768},
            )
            if not is_patchright:
                await ctx.add_init_script(_STEALTH_INIT_SCRIPT)
            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            solved = False
            for i in range(wait_s):
                await asyncio.sleep(1)
                title = await page.title()
                content_len = len(await page.content())
                cookies = await ctx.cookies()
                cf = next((c for c in cookies if c["name"] == "cf_clearance"), None)
                if cf and not _is_challenge(title, content_len):
                    logger.info("challenge cleared after %ds (title=%r, %d bytes)",
                                i + 1, title[:40], content_len)
                    solved = True
                    break
            cookies = await ctx.cookies()
            ua = await page.evaluate("navigator.userAgent")
            has_cf = any(c["name"] == "cf_clearance" for c in cookies)
            return cookies, ua, (solved and has_cf)
        finally:
            await browser.close()


def _cookie_header(cookies: list[dict]) -> str:
    # domain cookies for thuvienphapluat.vn, in a stable order
    parts = [f"{c['name']}={c['value']}" for c in cookies if "thuvienphapluat" in c.get("domain", "")]
    return "; ".join(parts)


def mint(url: str, out_cookie: Path, out_ua: Path, *, headless: bool = True, wait_s: int = 45) -> bool:
    cookies, ua, ok = asyncio.run(_mint(url, headless=headless, wait_s=wait_s))
    header = _cookie_header(cookies)
    if not ok or "cf_clearance=" not in header:
        logger.error("mint failed: cf_clearance not obtained (got %d cookies)", len(cookies))
        return False
    out_cookie.write_text(header, encoding="utf-8")
    os.chmod(out_cookie, 0o600)
    if ua:
        out_ua.write_text(ua, encoding="utf-8")
    logger.info("wrote %d-byte cookie header -> %s (UA=%s)", len(header), out_cookie, ua)
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Mint a fresh thuvienphapluat cf_clearance on the DGX.")
    p.add_argument("--url", default=DEFAULT_CHALLENGE_URL)
    p.add_argument("--out-cookie", type=Path, default=Path("~/.tvpl_cookie").expanduser())
    p.add_argument("--out-ua", type=Path, default=Path("~/.tvpl_ua").expanduser())
    p.add_argument("--headed", action="store_true", help="headed under a display/xvfb (stealthier)")
    p.add_argument("--wait", type=int, default=45, help="seconds to wait for the challenge to clear")
    args = p.parse_args(argv)
    ok = mint(args.url, args.out_cookie.expanduser(), args.out_ua.expanduser(),
              headless=not args.headed, wait_s=args.wait)
    print("MINTED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
