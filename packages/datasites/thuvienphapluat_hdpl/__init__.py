"""thuvienphapluat.vn / hoi-dap-phap-luat (legal Q&A) datasite.

A lightweight HTML crawler for thuvienphapluat.vn's "Hỏi đáp pháp luật"
(legal question-and-answer) section. Unlike the PDF datasites, this is a
pure HTML crawl behind Cloudflare: a warm :class:`requests.Session` with
browser-shaped headers passes the WAF; occasional rate-based 403 /
"Just a moment" challenges are ridden out with a cool-down + retry. An
optional subscriber cookie (a logged-in thuvienphapluat.vn account)
unlocks gated content and steadier access.

    python -m packages.datasites.thuvienphapluat_hdpl.crawl --output ~/data
    python -m packages.datasites.thuvienphapluat_hdpl.hf_export --push
"""
