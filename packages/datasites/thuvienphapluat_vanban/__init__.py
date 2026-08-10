"""thuvienphapluat.vn legal-document *catalog* crawler (cong-van / TCVN / van-ban).

The document search endpoint ``/page/tim-van-ban.aspx?type=N`` passes
Cloudflare (with a logged-in cookie), but the individual document pages
(``/cong-van/…​.aspx`` etc.) are hard-blocked. So this crawler harvests
the **catalog** (title, doc number, category, id, URL) from the paginated
search results -- a searchable index of the corpus, without full text.

    python -m packages.datasites.thuvienphapluat_vanban.catalog \
        --cookie-file ~/.tvpl_cookie --ua-file ~/.tvpl_ua --output ~/data
"""
