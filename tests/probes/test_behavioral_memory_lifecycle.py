import json

import pytest

from capclaims.behavioral_memory.lifecycle import freeze_preview, verify_preview


def sample_cases():
    people = []
    for i in range(18):
        people.append({"person_id": f"person-{i}",
            "history": [{"round": r, "p": 100, "Q": 10, "D": 9,
                "orders": [{"side": "buy", "qty": 1, "price": 99}], "profit_cents": 1234}
                for r in range(6, 13)],
            "current_state": {"round": 13, "p": 100, "w": 10, "Q": 10, "D": 9},
            "label": {"orders": [], "profit_cents": 0}})
    return people


def test_preview_seals_separate_labels_and_is_replay_validated(tmp_path):
    result = freeze_preview(sample_cases(), {"source_sha256": "a" * 64,
        "source_policy_verified": True}, tmp_path / "preview")
    assert result["trajectory_n"] == 108
    assert result["preview_sha256"]
    assert (tmp_path / "preview" / "scoring-labels.json").exists()
    public = json.loads((tmp_path / "preview" / "preview.json").read_text(encoding="utf-8"))
    assert "participant-private" not in json.dumps(public)
    assert '"labels"' not in json.dumps(public)
    assert verify_preview(tmp_path / "preview", result["preview_sha256"])["status"] == "PREVIEW_VALID"


def test_preview_rejects_missing_or_noncontiguous_formal_history(tmp_path):
    cases = sample_cases()
    cases[0]["history"].pop()
    with pytest.raises(ValueError):
        freeze_preview(cases, {"source_sha256": "a" * 64,
            "source_policy_verified": True}, tmp_path / "bad")
