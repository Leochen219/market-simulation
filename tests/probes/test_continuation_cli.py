"""The CLI must never print a private preview manifest."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def test_preview_and_verify_stdout_exclude_private_manifest(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "tools/probes/jev_continuation.py"
    spec = importlib.util.spec_from_file_location("continuation_cli_privacy_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    manifest = {"preview_sha256": "a" * 64, "sessions": [{"person_id": "PRIVATE_PERSON_DO_NOT_PRINT"}] * 24,
                "preflight_max_body_bytes": 6000, "raw_history": "PRIVATE_HISTORY_DO_NOT_PRINT"}
    fake = SimpleNamespace(build_preview=lambda *args: manifest, verify_preview=lambda *args: manifest)
    import capclaims
    monkeypatch.setitem(sys.modules, "capclaims.continuation_study", fake)
    monkeypatch.setattr(capclaims, "continuation_study", fake, raising=False)
    for argv in (["probe", "preview", "parent", "out"],
                 ["probe", "verify-preview", "out", "--reviewed-sha256", "a" * 64]):
        monkeypatch.setattr(sys, "argv", argv)
        assert cli.main() == 0
        out = capsys.readouterr().out
        assert "PRIVATE_PERSON" not in out and "PRIVATE_HISTORY" not in out
        assert json.loads(out)["session_n"] == 24


def test_replay_reports_verified_portfolio_count_without_private_orders(monkeypatch, capsys):
    import capclaims
    path = Path(__file__).resolve().parents[2] / "tools/probes/jev_continuation.py"
    spec = importlib.util.spec_from_file_location("continuation_cli_replay_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fake = SimpleNamespace(replay_run=lambda *args: {"session_n": 24,
        "portfolios": {str(i): "PRIVATE_ORDER_DO_NOT_PRINT" for i in range(24)}})
    monkeypatch.setitem(sys.modules, "capclaims.continuation_study", fake)
    monkeypatch.setattr(capclaims, "continuation_study", fake, raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "replay", "preview", "run"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert json.loads(out)["complete_n"] == 24 and "PRIVATE_ORDER" not in out
