"""User-visible CLI output must not expose a private manifest or score rows."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def _cli(monkeypatch, fake):
    path = Path(__file__).resolve().parents[2] / "tools/probes/jev_order_explanation.py"
    spec = importlib.util.spec_from_file_location("order_explanation_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import capclaims
    monkeypatch.setitem(sys.modules, "capclaims.order_explanation_study", fake)
    monkeypatch.setattr(capclaims, "order_explanation_study", fake, raising=False)
    return module


def test_cli_preview_and_replay_only_show_safe_counts(monkeypatch, capsys):
    secret = "PRIVATE_PARTICIPANT_DO_NOT_PRINT"
    manifest = {"preview_sha256": "a" * 64, "sessions": [secret] * 24,
        "preflight_max_body_bytes": 7000, "private_history": secret}
    fake = SimpleNamespace(build_preview=lambda *args: manifest,
        replay_run=lambda *args: {"session_n": 24, "portfolios": {str(i): secret for i in range(24)}})
    cli = _cli(monkeypatch, fake)
    for command in ("preview", "replay"):
        monkeypatch.setattr(sys, "argv", ["probe", command, "private-source", "private-output"])
        assert cli.main() == 0
        text = capsys.readouterr().out
        assert secret not in text
        assert json.loads(text)["session_n"] == 24


def test_cli_score_rejects_private_top_level_fields(monkeypatch, capsys):
    secret = "PRIVATE_SCORE_DO_NOT_PRINT"
    fake = SimpleNamespace(score_run=lambda *args: {"status": "VALID", "n_people": 6,
        "private_case_scores": [secret], "private_person_scores": [secret], "unexpected": secret,
        "paired_treatment_minus_control": {"mean": 0.0}})
    cli = _cli(monkeypatch, fake)
    monkeypatch.setattr(sys, "argv", ["probe", "score", "private-source", "private-output"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert secret not in output
    assert json.loads(output)["n_people"] == 6
