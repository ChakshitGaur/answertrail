import os

from config.env import load_dotenv


def test_parses_comments_quotes_export_and_inline_comments(tmp_path, monkeypatch):
    for key in ("A_KEY", "B_KEY", "C_KEY", "D_KEY", "E_KEY", "F_KEY"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "A_KEY=plain\n"
        'B_KEY="double quoted # not a comment"\n'
        "C_KEY='single'\n"
        "export D_KEY=exported\n"
        "E_KEY=value # trailing comment\n"
        "F_KEY=https://example.com/a?b=c\n"
        "not a valid line\n",
        encoding="utf-8",
    )
    load_dotenv(env)
    assert os.environ["A_KEY"] == "plain"
    assert os.environ["B_KEY"] == "double quoted # not a comment"
    assert os.environ["C_KEY"] == "single"
    assert os.environ["D_KEY"] == "exported"
    assert os.environ["E_KEY"] == "value"
    assert os.environ["F_KEY"] == "https://example.com/a?b=c"


def test_real_environment_wins_and_empty_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP_ME", "from-shell")
    monkeypatch.delenv("EMPTY_ONE", raising=False)
    env = tmp_path / ".env"
    env.write_text("KEEP_ME=from-file\nEMPTY_ONE=\n", encoding="utf-8")
    load_dotenv(env)
    assert os.environ["KEEP_ME"] == "from-shell"
    assert "EMPTY_ONE" not in os.environ


def test_missing_file_and_utf8_bom_are_fine(tmp_path, monkeypatch):
    load_dotenv(tmp_path / "does-not-exist.env")  # must not raise
    monkeypatch.delenv("BOM_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfBOM_KEY=ok\n")  # Notepad on Windows can save with a BOM
    load_dotenv(env)
    assert os.environ["BOM_KEY"] == "ok"
