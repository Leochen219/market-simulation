import pytest
import hashlib
import io
import zipfile
from types import SimpleNamespace

from capclaims.personalization import Episode, SourceManifest, Target, ParticipantConditioner
from capclaims.backends.jev import JevBackend
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest
from capclaims.personalization_xlsx import inspect_workbook, load_paper_episodes


def source():
    return SourceManifest("synthetic", "demo-v1", "0" * 64, True, True, True)


def episode(person, round_no, q, *, phase="procurement", scope="sheet-a", treatment="NV", demand=None):
    observed = {"p": 80, "w": 10, "Q": q}
    if phase != "procurement":
        observed["D"] = 50 if demand is None else demand
        return Episode(person, scope, treatment, round_no, phase, observed, {"orders": []})
    return Episode(person, scope, treatment, round_no, phase, {"p": observed["p"], "w": observed["w"]}, {"Q": q})


def target(round_no, *, phase="procurement", scope="sheet-a", treatment="NV"):
    return Target("request-1", "agent-1", "person-a", scope, treatment, round_no, phase, "decision-1")


def test_selects_latest_three_strict_prior_rounds_for_bound_person_and_phase():
    records = [episode("person-a", n, n) for n in range(1, 6)]
    records += [episode("person-b", 5, 99), episode("person-a", 6, 98, phase="orders")]
    ctx = ParticipantConditioner(source(), records).prepare(target(5))
    assert [row["action"]["Q"] for row in ctx["episodes"]] == [2, 3, 4]
    assert "D" not in str(ctx)
    assert "person-a" not in str(ctx) and "agent-1" not in str(ctx)


def test_person_agent_binding_and_freeze_fail_closed():
    c = ParticipantConditioner(source(), [episode("person-a", 1, 4)])
    c.prepare(target(2))
    with pytest.raises(ValueError, match="binding"):
        c.prepare(Target("request-2", "agent-1", "person-b", "sheet-a", "NV", 2,
                         "procurement", "decision-2"))
    with pytest.raises(ValueError, match="frozen"):
        c.prepare(Target("request-3", "agent-1", "person-a", "sheet-a", "NV", 3,
                         "procurement", "decision-1"))


def test_unknown_source_binding_fails_but_known_person_can_have_no_prior_history():
    c = ParticipantConditioner(source(), [episode("person-a", 5, 10)])
    with pytest.raises(ValueError, match="no source participant"):
        c.prepare(Target("r0", "a0", "missing-person", "sheet-a", "NV", 1,
                         "procurement", "decision-0"))
    context = c.prepare(Target("r1", "a1", "person-a", "sheet-a", "NV", 1,
                               "procurement", "decision-1"))
    assert context["episodes"] == []


def test_human_export_gates_are_off_by_default():
    manifest = SourceManifest("human", "paper-workbook", "0" * 64, True, False, False)
    c = ParticipantConditioner(manifest, [episode("p", 1, 1)])
    with pytest.raises(PermissionError, match="attestation"):
        c.prepare(Target("r", "a", "p", "sheet-a", "NV", 2, "procurement", "k"), provider=True)


def test_context_is_invariant_to_person_scope_and_source_checksum_labels():
    a = ParticipantConditioner(SourceManifest("synthetic", "a", "0" * 64, True, True, True),
        [episode("person-a", 1, 10, scope="scope-a")]).prepare(
            Target("r1", "agent-1", "person-a", "scope-a", "NV", 2, "procurement", "key-a"))
    b = ParticipantConditioner(SourceManifest("synthetic", "b", "1" * 64, True, True, True),
        [episode("person-b", 1, 10, scope="scope-b")]).prepare(
            Target("r2", "agent-2", "person-b", "scope-b", "NV", 2, "procurement", "key-b"))
    assert {k: v for k, v in a.items() if k != "context_sha256"} == {k: v for k, v in b.items() if k != "context_sha256"}
    assert a["context_sha256"] == b["context_sha256"]
    assert not any(secret in str(a) for secret in ("person-a", "agent-1", "scope-a", "0" * 64))


@pytest.mark.parametrize("observed,action", [
    ({"p": 80, "w": 10, "D": 30}, {"Q": 20}),
    ({"p": 80, "w": 10}, {"Q": "free text"}),
    ({"p": True, "w": 10}, {"Q": 20}),
    ({"p": 80, "w": 10.5}, {"Q": 20}),
])
def test_procurement_episode_rejects_unknown_or_invalid_fields(observed, action):
    with pytest.raises(ValueError):
        Episode("p", "s", "NV", 1, "procurement", observed, action)


def test_orders_episode_supports_g3_portfolios_but_enforces_sell_cap_and_snapshot_immutability():
    observed = {"p": 80, "w": 50, "Q": 10, "D": 100}
    action = {"orders": [{"type": "SELL", "amount": 8, "price": 999}]}
    ep = Episode("p", "s", "G3", 1, "orders", observed, action)
    observed["Q"] = 400
    action["orders"][0]["amount"] = 1
    assert ep.observed["Q"] == 10 and ep.action["orders"][0]["amount"] == 8
    with pytest.raises(ValueError, match="sell cap"):
        Episode("p", "s", "G3", 1, "orders", {"p": 80, "w": 50, "Q": 10, "D": 100},
            {"orders": [{"type": "SELL", "amount": 11, "price": 50}]})


def _column_name(index):
    value = ""
    while index:
        index, rem = divmod(index - 1, 26)
        value = chr(65 + rem) + value
    return value


def _paper_zip_fixture(path):
    nv_headers = ["Player", "Round", "Key1", "Order", "Sales Price", "Demand", "Profit ($)",
                  "Excess Initial (#)", "Shortage Initial (#)"]
    market_headers = ["Player", "Round", "PR Key", "Room", "Sales Price", "Purchase Cost", "Order",
        "Actual Demand", "Sales (#)", "Profit Projection ()", "Sales New (#)", "Excess Initial (#)",
        "Shortage Initial (#)", "Trade Cost", "Post-trade profit ()", "Total profit (cum) ()",
        "Units sold (#) (hidden)", "Units bought (#) (hidden)", "Excess Final (#)", "Shortage Final (#)"]
    market_headers += [f"{field}_TR{i}" for i in range(1, 6) for field in ("Type", "Amount", "Price")]
    all_strings = list(dict.fromkeys(nv_headers + market_headers + ["", "Player", "Key", "Room", "Buy ", "Sell"] +
        [f"fixture-player-{i}" for i in range(1, 7)]))
    string_idx = {value: i for i, value in enumerate(all_strings)}
    shared = "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">" + "".join(
        f"<si><t>{value}</t></si>" for value in all_strings) + "</sst>"
    names = ["NV 50", "G3 50", "GA 50", "NV 10", "G3 10", "GA 10"]
    sheet_xml = {}
    for sheet_i, name in enumerate(names, 1):
        headers = nv_headers if name.startswith("NV") else market_headers
        row = {"Player": f"fixture-player-{sheet_i}", "Round": 1, "Key1": "Key", "Order": 10,
               "Sales Price": 80, "Demand": 25, "Profit ($)": 10, "Excess Initial (#)": 0,
               "Shortage Initial (#)": 0, "PR Key": "Key", "Room": "Room", "Purchase Cost": 50,
               "Actual Demand": 100, "Sales (#)": 0, "Profit Projection ()": 0, "Sales New (#)": 0,
               "Trade Cost": 0, "Post-trade profit ()": 0, "Total profit (cum) ()": 0,
               "Units sold (#) (hidden)": 0, "Units bought (#) (hidden)": 0,
               "Excess Final (#)": 0, "Shortage Final (#)": 0,
               "Type_TR1": "Buy ", "Amount_TR1": 3, "Price_TR1": 70,
               "Type_TR2": "Sell", "Amount_TR2": 2, "Price_TR2": 85}
        cells = []
        for line, values in enumerate((headers, [row.get(header, "") for header in headers]), 1):
            for col, value in enumerate(values, 1):
                ref = f"{_column_name(col)}{line}"
                if isinstance(value, str):
                    cells.append(f"<c r=\"{ref}\" t=\"s\"><v>{string_idx[value]}</v></c>")
                elif value != "":
                    cells.append(f"<c r=\"{ref}\"><v>{value}</v></c>")
        row1, row2 = cells[:len(headers)], cells[len(headers):]
        sheet_xml[f"xl/worksheets/sheet{sheet_i}.xml"] = (
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>"
            f"<row r=\"1\">{''.join(row1)}</row><row r=\"2\">{''.join(row2)}</row></sheetData></worksheet>")
    workbook = ("<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>" +
        "".join(f"<sheet name=\"{name}\" sheetId=\"{i}\" r:id=\"rId{i}\"/>" for i, name in enumerate(names, 1)) +
        "</sheets></workbook>")
    rels = ("<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" +
        "".join(f"<Relationship Id=\"rId{i}\" Target=\"worksheets/sheet{i}.xml\" "
                "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\"/>"
                for i in range(1, 7)) + "</Relationships>")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as workbook_zip:
        workbook_zip.writestr("xl/sharedStrings.xml", shared)
        workbook_zip.writestr("xl/workbook.xml", workbook)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", rels)
        for member, content in sheet_xml.items():
            workbook_zip.writestr(member, content)
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("Data_vf.xlsx", buffer.getvalue())


def test_local_xlsx_adapter_maps_six_sheets_without_outcomes_or_execution_data(tmp_path):
    path = tmp_path / "paper.zip"
    _paper_zip_fixture(path)
    with zipfile.ZipFile(path) as outer:
        inner = outer.read("Data_vf.xlsx")
    direct_path = tmp_path / "Data_vf.xlsx"
    direct_path.write_bytes(inner)
    outer_metadata = inspect_workbook(path)
    direct_metadata = inspect_workbook(direct_path)
    assert outer_metadata["sheets"] == direct_metadata["sheets"]
    outer_manifest, outer_episodes = load_paper_episodes(path)
    manifest, episodes = load_paper_episodes(direct_path)
    assert outer_episodes == episodes
    assert outer_manifest.source_sha256 == manifest.source_sha256
    assert len(episodes) == 10
    assert sum(e.phase == "procurement" for e in episodes) == 6
    assert sum(e.phase == "orders" for e in episodes) == 4
    assert manifest.source_sha256 == hashlib.sha256(inner).hexdigest()
    assert manifest.intended_use_authorized is False and manifest.provider_transmission_permitted is False
    market_orders = [e for e in episodes if e.phase == "orders"]
    assert {(e.treatment, e.observed["w"]) for e in market_orders} == {("G3", 50), ("GA", 50), ("G3", 10), ("GA", 10)}
    assert market_orders[0].action["orders"] == ({"type": "BUY", "amount": 3, "price": 70},
        {"type": "SELL", "amount": 2, "price": 85})
    assert all(set(e.observed) == {"p", "w"} for e in episodes if e.phase == "procurement")


def test_jev_prepares_private_history_into_state_once_and_keeps_original_reply_envelope():
    async def run():
        class Client:
            calls = []
            async def system_one(self, state, questions, **kwargs):
                self.calls.append((state, questions))
                ids = list(questions["decision"].criteria)
                return SimpleNamespace(model="jev-1.14.0", choices={"decision": SimpleNamespace(
                    probabilities={key: (1.0 if i == 0 else 0.0) for i, key in enumerate(ids)})}, usage={})
            async def close(self): pass
        request = DecisionRequest("agent-1", 4, "paper:purchase:h", {"decision_interface": "economic_v2",
            "decision_field": "purchase", "choice_minimum": 0, "choice_maximum": 400},
            tuple(Candidate(f"d{i}", i, str(i)) for i in range(2)))
        conditioner = ParticipantConditioner(source(), [episode("person-a", 4, 33)])
        c = Client()
        be = JevBackend(api_key="fake", rps=9999, tps=10_000_000, max_requests=1,
            max_estimated_input_tokens=100000, client=c, instruction_profile="participant_prediction_v1",
            objective="participant_prediction_v1", participant_conditioner=conditioner,
            participant_targets={request.request_id: Target(request.request_id, request.agent_id, "person-a",
                "sheet-a", "NV", 5, "procurement", "decision-1")}, load_environment=False)
        (reply,) = await asyncio.wait_for(be.decide_many([request]), 2)
        state, _ = c.calls[0]
        assert state["participant_history"]["episodes"][0]["action"] == {"Q": 33}
        assert reply.observation_hash == request.observation_hash
        assert reply.timing["participant_history_episode_count"] == 1
        assert len(reply.timing["participant_history_sha256"]) == 64
        assert len(reply.timing["effective_state_sha256"]) == 64
        assert "context_sha256" not in state["participant_history"]
        assert "participant_history" not in reply.timing
    import asyncio
    asyncio.run(run())


@pytest.mark.parametrize("observation", [
    {"decision_field": "purchase", "D": 75},
    {"decision_field": "purchase", "participant_history": {"injected": True}},
])
def test_jev_rejects_realized_procurement_demand_or_reserved_context_before_any_call(observation):
    import asyncio
    class Client:
        calls = []
        async def system_one(self, *args, **kwargs): self.calls.append(args)
    client = Client()
    request = DecisionRequest("agent-1", 4, "paper:purchase:h", observation,
        (Candidate("d0", 0, "zero"), Candidate("d1", 1, "one")))
    target = Target(request.request_id, request.agent_id, "person-a", "sheet-a", "NV", 5,
        "procurement", "decision-1")
    be = JevBackend(api_key="fake", rps=9999, tps=10_000_000, max_requests=2,
        max_estimated_input_tokens=100000, client=client, instruction_profile="participant_prediction_v1",
        objective="participant_prediction_v1", participant_conditioner=ParticipantConditioner(source(), []),
        participant_targets={request.request_id: target}, load_environment=False)
    async def run():
        with pytest.raises(ValueError):
            await be.decide_many([request])
        assert client.calls == [] and be.budget_usage["requests_reserved"] == 0
    asyncio.run(run())


def test_jev_participant_batch_preflight_sends_nothing_when_any_target_is_missing():
    import asyncio
    class Client:
        calls = []
        async def system_one(self, *args, **kwargs): self.calls.append(args)
    c = Client()
    request = DecisionRequest("agent-1", 5, "paper:purchase:h", {"decision_interface": "economic_v2",
        "decision_field": "purchase"}, (Candidate("d0", 0, "zero"), Candidate("d1", 1, "one")))
    be = JevBackend(api_key="fake", rps=9999, tps=10_000_000, max_requests=2,
        max_estimated_input_tokens=100000, client=c, instruction_profile="participant_prediction_v1",
        objective="participant_prediction_v1", participant_conditioner=ParticipantConditioner(source(), []),
        load_environment=False)
    async def run():
        with pytest.raises(ValueError, match="target"):
            await be.decide_many([request])
        assert not c.calls
        assert be.budget_usage["requests_reserved"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("treatment,phase", [("NV", "procurement"), ("NV", "orders"),
                                             ("GA", "procurement"), ("GA", "orders")])
def test_real_paper_observations_use_neutral_profit_accounting_for_prediction(treatment, phase):
    import asyncio
    from capclaims.paper.observations import build_paper_observation
    from capclaims.engine.protocol import validate_reply

    field = "purchase" if phase == "procurement" else "quantity"
    stage = "paper:purchase:h" if phase == "procurement" else "paper:slot0:qty:h"
    observation = build_paper_observation(agent_id="demo-agent", round_index=0, treatment=treatment,
        n_agents=10, wholesale_price=50 if treatment == "GA" else 10, p=80, history=(), partial_orders=(),
        phase=phase, q=10 if phase == "orders" else None, d=40 if phase == "orders" else None,
        slot=0 if phase == "orders" else None, decision_field=field, minimum=0, maximum=400,
        decision_interface="economic_v2")
    original = dict(observation)
    request = DecisionRequest("demo-agent", 0, stage, observation,
        tuple(Candidate(f"digit{i}", i, str(i)) for i in range(2)))

    class Client:
        calls = []
        async def system_one(self, state, questions, **kwargs):
            self.calls.append(state)
            ids = list(questions["decision"].criteria)
            return SimpleNamespace(model="jev-1.14.0", choices={"decision": SimpleNamespace(
                probabilities={key: (1.0 if i == 0 else 0.0) for i, key in enumerate(ids)})}, usage={})
        async def close(self): pass
    client = Client()
    backend = JevBackend(api_key="fake", rps=9999, tps=10_000_000, max_requests=1,
        max_estimated_input_tokens=100000, client=client, instruction_profile="participant_prediction_v1",
        objective="participant_prediction_v1", load_environment=False)
    async def run():
        (reply,) = await asyncio.wait_for(backend.decide_many([request]), 2)
        state = client.calls[0]
        profit = state["public_instructions"]["profit"]
        assert "Maximize" not in profit
        assert "profit accounting" in profit.lower()
        assert "sales revenue" in profit.lower()
        assert "wholesale cost of q" in profit.lower()
        assert "expires" in profit.lower()
        assert observation == original and observation["public_instructions"]["profit"].startswith("Maximize")
        validate_reply(request, reply)
        assert reply.observation_hash == request.observation_hash
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["paper:gate:purchase:h:economics", "paper:purchase:h"])
def test_diagnostic_paper_stage_is_rejected_by_backend_and_instruction_factory_before_dispatch(stage):
    import asyncio
    from capclaims.backends.jev_instructions import build_instruction
    request = DecisionRequest("demo-agent", 0, stage,
        {"decision_field": "purchase", "phase": "procurement", "diagnostic_task": "maximize profit"},
        (Candidate("d0", 0, "zero"), Candidate("d1", 1, "one")))
    with pytest.raises(ValueError, match="standard paper|diagnostic"):
        build_instruction(request, "participant_prediction_v1")

    class Client:
        calls = []
        async def system_one(self, *args, **kwargs): self.calls.append(args)
    client = Client()
    backend = JevBackend(api_key="fake", rps=9999, tps=10_000_000, max_requests=1,
        max_estimated_input_tokens=100000, client=client, instruction_profile="participant_prediction_v1",
        objective="participant_prediction_v1", load_environment=False)
    async def run():
        with pytest.raises(ValueError, match="standard paper|diagnostic"):
            await backend.decide_many([request])
        assert not client.calls and backend.budget_usage["requests_reserved"] == 0
    asyncio.run(run())
