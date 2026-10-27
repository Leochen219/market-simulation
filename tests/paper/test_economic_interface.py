import pytest

from capclaims.paper.interface import build_candidates, numeric_observation
from capclaims.paper.codec import prefix_mask
from capclaims.paper.config import PaperConfig
from capclaims.paper.observations import build_paper_observation


def test_legacy_candidates_keep_existing_ids_values_and_labels():
    obs = {"decision_interface": "legacy_v1"}
    rows = build_candidates((0, 1, 2), obs)
    assert [(r.candidate_id, r.value, r.label) for r in rows] == [
        ("c0", 0, "0"), ("c1", 1, "1"), ("c2", 2, "2")]


def test_economic_candidates_explain_full_feasible_branch_and_exact_last_digit():
    obs = {"decision_interface": "economic_v2", "numeric_prefix": "",
           "choice_minimum": 0, "choice_maximum": 400, "decision_field": "purchase"}
    rows = build_candidates(prefix_mask("", 0, 400), obs)
    assert rows[3].value == 3
    assert "300..399" in rows[3].label and "units" in rows[3].label

    prefixed = {**obs, "numeric_prefix": "17"}
    row = next(r for r in build_candidates(prefix_mask("17", 0, 400), prefixed) if r.value == 3)
    assert "173" in row.label and "units" in row.label


def test_economic_candidates_cover_singleton_400_and_leading_zero_bounds():
    row = build_candidates((0,), {"decision_interface": "economic_v2", "numeric_prefix": "40",
        "choice_minimum": 0, "choice_maximum": 400, "decision_field": "purchase"})[0]
    assert row.value == 0 and "400" in row.label and "units" in row.label
    leading_zero = build_candidates((0,), {"decision_interface": "economic_v2", "numeric_prefix": "00",
        "choice_minimum": 0, "choice_maximum": 400, "decision_field": "purchase"})[0]
    assert "0" in leading_zero.label and "units" in leading_zero.label


def test_numeric_observation_has_full_decimal_field_name_and_units():
    obs = numeric_observation({"phase": "orders", "decision_field": "quantity",
                               "decision_interface": "economic_v2"}, "17", 1, 999, "u")
    assert obs["numeric_prefix"] == "17"
    assert obs["choice_minimum"] == 1 and obs["choice_maximum"] == 999
    assert obs["decimal_place"] == "u"
    assert obs["decimal_place_name"] == "ones"
    assert obs["numeric_value_name"] == "order quantity"
    assert obs["numeric_units"] == "units"
    from capclaims.backends.laya import production_instruction
    assert "three digits" in production_instruction("paper:purchase:h", "economic_v2")
    assert "leading zeros count" in production_instruction("paper:purchase:h", "economic_v2")


def test_economic_interface_is_opt_in_and_legacy_observation_is_unchanged():
    assert PaperConfig().decision_interface == "legacy_v1"
    legacy = build_paper_observation(agent_id="a", round_index=0, treatment="GA", n_agents=10,
        wholesale_price=10, p=20, history=(), partial_orders=(), phase="procurement")
    assert "decision_interface" not in legacy
    modern = build_paper_observation(agent_id="a", round_index=0, treatment="GA", n_agents=10,
        wholesale_price=10, p=20, history=(), partial_orders=(), phase="procurement",
        decision_interface="economic_v2")
    assert modern["decision_interface"] == "economic_v2"
    assert "per-round expected profit" in modern["public_instructions"]["profit"]
    assert "carryover" in modern["public_instructions"]["profit"]


def test_economic_side_candidates_have_explicit_action_meanings():
    rows = build_candidates(("buy", "STOP", "sell"),
        {"decision_interface": "economic_v2", "decision_field": "side"})
    assert [(r.candidate_id, r.value) for r in rows] == [
        ("action-buy", "buy"), ("action-STOP", "STOP"), ("action-sell", "sell")]
    assert all(r.label for r in rows)


def test_laya_preflight_rejects_compound_head_overflow(monkeypatch):
    import sys
    import types

    from capclaims.backends.laya import _preflight_laya_question

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    def build_sequence(_tok, _state, question, **kwargs):
        instruction = question["t"] + " question: " + question["ins"]
        sequence = [101, *range(len(instruction.split())), 102]
        return (sequence, list(range(1, 11)),
            {"options": 10, "options_distinct": 10, "tokens_per_option": 16},
            {"state_tokens_dropped": 0})

    render_options = lambda question: [f"{key}: {value}" for key, value in question["crit"].items()]
    monkeypatch.setitem(sys.modules, "laya.common", types.SimpleNamespace(
        build_sequence=build_sequence, render_options=render_options))
    agent = types.SimpleNamespace(tok=Tokenizer())
    question = {"decision": {"instructions": "Short instruction.",
        "criteria": {f"digit{i}": "words words" for i in range(10)}}}
    try:
        _preflight_laya_question(agent, "{}", question, list(question["decision"]["criteria"].values()))
    except ValueError as exc:
        assert "cap economic candidate labels" in str(exc)
    else:
        raise AssertionError("Laya preflight accepted a choice head whose combined options were clipped")


def test_economic_nv_session_replays_strict_numeric_semantics_and_rejects_rehashed_label_tamper(tmp_path):
    import asyncio
    import json

    from capclaims.engine.protocol import DecisionReply, digest_json
    from capclaims.paper.replay import replay_paper
    from capclaims.paper.session import run_paper_session

    class FirstChoiceBackend:
        async def decide_many(self, requests):
            replies = []
            for request in requests:
                probabilities = {c.candidate_id: float(i == 0)
                                 for i, c in enumerate(request.candidates)}
                replies.append(DecisionReply.from_request(request, probabilities, probabilities))
            return replies

    config = PaperConfig(treatment="NV", n_agents=2, warmup_rounds=0, analyzed_rounds=1,
                         decision_interface="economic_v2")
    path = tmp_path / "economic-v2"
    result = asyncio.run(run_paper_session(config, FirstChoiceBackend(), path))
    replay = replay_paper(path)
    assert replay.final_hash == result.final_hash
    audits_path = path / "audit.jsonl"
    rows = [json.loads(line) for line in audits_path.read_text(encoding="utf-8").splitlines()]
    item = next(row for row in rows if row["agent_id"] == "agent_00" and row["stage"] == "paper:purchase:h")
    item["candidates"][3]["label"] = "digit 3: exact resulting procurement quantity 333 units"
    item["candidate_set_hash"] = digest_json(item["candidates"])
    rows[rows.index(next(row for row in rows if row["request_id"] == item["request_id"]))] = item
    audits_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                           encoding="utf-8")
    try:
        replay_paper(path)
    except ValueError as exc:
        assert "observation or legal candidates" in str(exc)
    else:
        raise AssertionError("replay accepted a rehashed economic candidate-label tamper")


def test_economic_ga_buy_sell_partial_orders_replay_uses_shared_observation_schema(tmp_path):
    import asyncio

    from capclaims.engine.protocol import DecisionReply
    from capclaims.paper.replay import replay_paper
    from capclaims.paper.session import run_paper_session

    class BuySellBackend:
        async def decide_many(self, requests):
            replies = []
            for request in requests:
                if request.stage.endswith(":side"):
                    slot = int(request.stage.split(":slot", 1)[1].split(":", 1)[0])
                    value = "buy" if slot == 0 else "sell" if slot == 1 else "STOP"
                    selected = next(c for c in request.candidates if c.value == value)
                elif request.stage == "paper:purchase:h":
                    selected = next(c for c in request.candidates if c.value == 1)
                else:
                    selected = request.candidates[0]
                probabilities = {c.candidate_id: float(c == selected) for c in request.candidates}
                replies.append(DecisionReply.from_request(request, probabilities, probabilities))
            return replies

    config = PaperConfig(treatment="GA", n_agents=2, warmup_rounds=0, analyzed_rounds=1,
                         decision_interface="economic_v2")
    path = tmp_path / "economic-v2-ga"
    result = asyncio.run(run_paper_session(config, BuySellBackend(), path))
    replay = replay_paper(path)
    assert replay.final_hash == result.final_hash
    orders = replay.events[0]["orders"]
    assert {order["side"] for order in orders} == {"buy", "sell"}
    assert all(sum(o["qty"] for o in orders if o["owner"] == agent and o["side"] == "sell") <= 100
               for agent in ("agent_00", "agent_01"))
