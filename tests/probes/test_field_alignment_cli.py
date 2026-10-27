"""Public CLI output contains counts and aggregates, never private preview data."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def load_cli():
    path = Path(__file__).resolve().parents[2] / "tools/probes/jev_field_alignment.py"
    spec = importlib.util.spec_from_file_location("field_alignment_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_preview_stdout_redacts_private_cases(monkeypatch, capsys):
    import capclaims
    manifest = {"preview_sha256": "a" * 64, "sessions": [{"person_id": "PRIVATE_SENTINEL"}] * 24,
                "preflight_max_body_bytes": 6000, "raw_history": "PRIVATE_SENTINEL"}
    fake = SimpleNamespace(build_preview=lambda *args: manifest)
    monkeypatch.setitem(sys.modules, "capclaims.field_alignment_study", fake)
    monkeypatch.setattr(capclaims, "field_alignment_study", fake, raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "preview", "parent", "output"])
    assert load_cli().main() == 0
    output = capsys.readouterr().out
    assert "PRIVATE_SENTINEL" not in output
    assert json.loads(output)["session_n"] == 24


def test_replay_stdout_redacts_private_portfolios(monkeypatch, capsys):
    import capclaims
    fake = SimpleNamespace(replay_run=lambda *args: {"session_n": 24,
        "portfolios": {str(i): "PRIVATE_SENTINEL" for i in range(24)}})
    monkeypatch.setitem(sys.modules, "capclaims.field_alignment_study", fake)
    monkeypatch.setattr(capclaims, "field_alignment_study", fake, raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "replay", "preview", "run"])
    assert load_cli().main() == 0
    output = capsys.readouterr().out
    assert "PRIVATE_SENTINEL" not in output
    assert json.loads(output)["complete_n"] == 24


def test_score_stdout_does_not_expose_private_internal_rows(monkeypatch, capsys):
    import capclaims
    fake = SimpleNamespace(score_run=lambda *args: {"status": "SCORED", "n_people": 6,
        "per_arm": {"control": {"mean_loss": 1.0}},
        "private_case_scores": [{"person_id": "PRIVATE_SENTINEL", "loss": 42}]})
    monkeypatch.setitem(sys.modules, "capclaims.field_alignment_study", fake)
    monkeypatch.setattr(capclaims, "field_alignment_study", fake, raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "score", "preview", "run"])
    assert load_cli().main() == 0
    output = capsys.readouterr().out
    assert "PRIVATE_SENTINEL" not in output and "private_case_scores" not in output
    assert json.loads(output)["n_people"] == 6
