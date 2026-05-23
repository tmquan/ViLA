"""Single-file multi-tab dashboard renderer."""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer


def render_dashboard(out_dir: Path, title: str) -> None:
    """Emit ``dashboard.html`` that iframes every viz file in ``out_dir``.

    All interpolations are escaped:

    * filenames -> ``html.escape`` for HTML attributes/text,
      ``json.dumps`` for JS string literals (handles quotes, backslashes,
      and ``</script>`` sequences).
    * ``title`` -> ``html.escape`` for HTML text contexts.
    """
    htmls = sorted(p.name for p in out_dir.glob("*.html") if p.name != "dashboard.html")
    safe_title = _html.escape(title, quote=True)
    tabs_parts: list[str] = []
    for name in htmls:
        js = json.dumps(name)
        txt = _html.escape(name, quote=True)
        tabs_parts.append(
            f'<div class="tab"><button onclick="show({js})">{txt}</button></div>'
        )
    tabs = "\n".join(tabs_parts)
    frames_parts: list[str] = []
    for i, name in enumerate(htmls):
        attr = _html.escape(name, quote=True)
        frames_parts.append(
            f'<iframe id="f_{i}" data-name="{attr}" src="{attr}" '
            f'style="width:100%;height:90vh;border:none;display:none"></iframe>'
        )
    frames = "\n".join(frames_parts)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{safe_title}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 0; }}
  .tabs {{ display:flex; flex-wrap:wrap; background:#222; }}
  .tab button {{ color:#fff; background:#333; border:none; padding:8px 12px;
                 margin:2px; cursor:pointer; border-radius:2px; }}
  .tab button:hover {{ background:#555; }}
  h1 {{ margin:10px 16px; }}
</style>
<script>
  function show(name) {{
    document.querySelectorAll('iframe').forEach(f => {{
      f.style.display = (f.dataset.name === name) ? 'block' : 'none';
    }});
  }}
  window.addEventListener('load', () => {{
    const first = document.querySelector('iframe');
    if (first) first.style.display = 'block';
  }});
</script>
</head><body>
<h1>{safe_title}</h1>
<div class="tabs">{tabs}</div>
{frames}
</body></html>"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


class DashboardRenderer(Renderer):
    """Always-regenerated ``dashboard.html`` iframing every other viz."""

    name = "dashboard"
    bucket = "misc"

    def render(
        self,
        df: pd.DataFrame,
        *,
        out_dir: Path,
        cfg: Any,
        onto: Ontology,
        slug: str,
        force: bool,
    ) -> int:
        render_dashboard(out_dir, str(cfg.visualizer.dashboard_title))
        return 1


__all__ = ["DashboardRenderer", "render_dashboard"]
