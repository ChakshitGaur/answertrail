import re
from pathlib import Path

from django.conf import settings
from rest_framework.test import APIClient

STATIC = Path(settings.BASE_DIR) / "docqa" / "static" / "docqa"


def test_home_page_serves_the_chat_ui():
    resp = APIClient().get("/")
    assert resp.status_code == 200
    html = resp.content.decode()
    for needle in ("docqa/lib.js", "docqa/app.js", "docqa/style.css", 'id="composer"', 'id="dropzone"'):
        assert needle in html


def test_page_has_no_inline_scripts_so_a_strict_csp_would_work():
    html = APIClient().get("/").content.decode()
    scripts = re.findall(r"<script\b([^>]*)>", html)
    assert scripts and all("src=" in attrs for attrs in scripts)
    assert " onclick=" not in html and "javascript:" not in html


def test_every_element_id_the_script_uses_exists_in_the_page():
    html = APIClient().get("/").content.decode()
    used = set(re.findall(r"\$\(\"([a-z-]+)\"\)", (STATIC / "app.js").read_text(encoding="utf-8")))
    declared = set(re.findall(r'\bid="([^"]+)"', html))
    assert used and used <= declared, f"missing ids: {sorted(used - declared)}"


def test_ui_upload_limit_matches_the_server_limit():
    js = (STATIC / "lib.js").read_text(encoding="utf-8")
    assert f"MAX_UPLOAD_BYTES = {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} * 1024 * 1024" in js
