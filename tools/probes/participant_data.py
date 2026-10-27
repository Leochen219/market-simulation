"""Offline, synthetic-only inspection tools for participant history conditioning."""
from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

from capclaims.personalization import Episode, ParticipantConditioner, SourceManifest, Target
from capclaims.personalization_xlsx import inspect_workbook


def synthetic_fixture() -> tuple[SourceManifest, list[Episode]]:
    manifest = SourceManifest("synthetic", "anonymous-demo-v1", "0" * 64, True, True, True)
    episodes = [Episode("demo-person-1", "demo-sheet", "NV", r, "procurement",
        {"p": 72 + r, "w": 10}, {"Q": 90 + r}) for r in range(1, 5)]
    return manifest, episodes


def preview_synthetic() -> dict:
    manifest, episodes = synthetic_fixture()
    target = Target("synthetic-request", "synthetic-agent", "demo-person-1", "demo-sheet",
        "NV", 5, "procurement", "synthetic-decision-5")
    context = ParticipantConditioner(manifest, episodes).prepare(target)
    return {"mode": "synthetic-preview", "source_kind": "synthetic", "context": context}


async def mock_smoke() -> dict:
    from capclaims.backends.jev import JevBackend
    from capclaims.domain.types import Candidate
    from capclaims.engine.protocol import DecisionRequest

    class MockClient:
        def __init__(self):
            self.states = []
            self.active = self.maximum = 0
        async def system_one(self, state, questions, **kwargs):
            self.states.append(state)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                # Force completion order away from request order to exercise reassembly.
                await asyncio.sleep((9 - int(state["probe_index"])) * 0.002)
                ids = list(questions["decision"].criteria)
                return SimpleNamespace(model="jev-1.14.0", choices={"decision": SimpleNamespace(
                    probabilities={key: (1.0 if i == 0 else 0.0) for i, key in enumerate(ids)})}, usage={})
            finally:
                self.active -= 1
        async def close(self):
            return None

    manifest = SourceManifest("synthetic", "anonymous-mock-v1", "0" * 64, True, True, True)
    episodes = [Episode(f"demo-person-{i}", "demo-sheet", "NV", 1, "procurement",
        {"p": 70 + i, "w": 10}, {"Q": 20 + i}) for i in range(10)]
    conditioner = ParticipantConditioner(manifest, episodes)
    requests, targets = [], {}
    for i in range(10):
        request = DecisionRequest(f"demo-agent-{i}", 2, "paper:purchase:h",
            {"decision_interface": "economic_v2", "decision_field": "purchase", "probe_index": i,
             "choice_minimum": 0, "choice_maximum": 400},
            tuple(Candidate(f"digit{d}", d, str(d)) for d in range(2)))
        requests.append(request)
        targets[request.request_id] = Target(request.request_id, request.agent_id, f"demo-person-{i}",
            "demo-sheet", "NV", 3, "procurement", f"demo-decision-{i}")
    client = MockClient()
    backend = JevBackend(api_key="offline-mock-only", rps=10000, tps=1_000_000_000,
        max_requests=10, max_estimated_input_tokens=100000, max_in_flight=10, client=client,
        instruction_profile="participant_prediction_v1", objective="participant_prediction_v1",
        participant_conditioner=conditioner, participant_targets=targets, load_environment=False)
    try:
        replies = await backend.decide_many(requests)
        assert [reply.request_id for reply in replies] == [request.request_id for request in requests]
        assert client.maximum == 10 and len(client.states) == 10
        assert all(state["participant_history"]["episodes"][0]["action"]["Q"] == 20 + state["probe_index"]
                   for state in client.states)
        assert all("participant_history" not in timing for timing in (r.timing for r in replies))
        return {"mode": "synthetic-sdk-mock", "requests": len(replies), "max_in_flight": client.maximum,
                "reply_order_preserved": True, "distinct_private_histories": len(client.states)}
    finally:
        await backend.aclose()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="read local workbook structure without exporting rows")
    inspect.add_argument("--zip", required=True, dest="archive", help="path to supplied paper ZIP")
    sub.add_parser("preview", help="render a synthetic example context")
    sub.add_parser("mock-smoke", help="run ten synthetic private requests against an offline mock")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        import zipfile
        with zipfile.ZipFile(args.archive) as archive:
            candidates = [name for name in archive.namelist() if name.endswith("Data_vf.xlsx")]
            if len(candidates) != 1:
                raise SystemExit("expected exactly one Data_vf.xlsx member")
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory(prefix="participant-inspect-") as directory:
                path = Path(directory) / "Data_vf.xlsx"
                path.write_bytes(archive.read(candidates[0]))
                result = inspect_workbook(path)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "preview":
        print(json.dumps(preview_synthetic(), indent=2, sort_keys=True))
    elif args.command == "mock-smoke":
        print(json.dumps(asyncio.run(mock_smoke()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
