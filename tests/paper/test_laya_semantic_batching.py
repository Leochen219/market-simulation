import asyncio
from concurrent.futures import ThreadPoolExecutor

from capclaims.backends import laya as laya_module
from capclaims.backends.laya import LayaBackend, _candidate_signature
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest


def _request(agent_id, candidates):
    return DecisionRequest(agent_id, 0, "paper:purchase:h", {"agent": agent_id}, tuple(candidates))


def test_laya_batches_only_identical_candidate_semantics_and_keeps_homogeneous_batch(monkeypatch):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    observed = []

    def fake_predict(stage, candidates, states):
        observed.append((stage, candidates, states))
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        probabilities = {key: 1 / len(candidate_ids) for key in candidate_ids}
        usage = {"truncated": False, "input_tokens": 1, "output_tokens": 0,
                 "state_tokens": 1, "state_tokens_dropped": 0, "truncated_questions": [], "options": []}
        return {"rows": [{"probabilities": probabilities, "raw_scores": probabilities, "usage": usage}
                          for _ in states],
                "forward_batch_sizes": [len(states)], "group_size": len(states),
                "elapsed_seconds": 0.0, "peak_cuda_bytes": None, "revision": "cpu-test",
                "sdk_revision": "cpu-test", "device": "cpu", "dtype": "float32"}

    monkeypatch.setattr(laya_module, "_predict_group", fake_predict)
    backend = object.__new__(LayaBackend)
    backend._closed = False
    backend._worker_busy = False
    backend._executor = ThreadPoolExecutor(max_workers=2)

    base = [Candidate(f"c{i}", i, str(i)) for i in range(9)]
    shifted = [Candidate(f"c{i}", i + 1, str(i + 1)) for i in range(9)]
    replies = asyncio.run(backend.decide_many([_request("a", base), _request("b", shifted)]))
    assert len(replies) == 2
    assert len(observed) == 2
    assert sorted(len(states) for _, _, states in observed) == [1, 1]
    backend._executor.shutdown(wait=True)

    observed.clear()
    backend = object.__new__(LayaBackend)
    backend._closed = False
    backend._worker_busy = False
    backend._executor = ThreadPoolExecutor(max_workers=2)
    replies = asyncio.run(backend.decide_many([_request(f"agent_{i:02d}", base) for i in range(10)]))
    assert len(replies) == 10
    assert len(observed) == 1
    assert len(observed[0][2]) == 10
    backend._executor.shutdown(wait=True)


def test_candidate_signature_includes_values_and_labels_not_only_ids():
    base = [Candidate(f"c{i}", i, str(i)) for i in range(9)]
    shifted = [Candidate(f"c{i}", i + 1, str(i + 1)) for i in range(9)]
    relabeled = [Candidate(f"c{i}", i, f"Option {i}") for i in range(9)]
    assert _candidate_signature(base) != _candidate_signature(shifted)
    assert _candidate_signature(base) != _candidate_signature(relabeled)
    assert _candidate_signature(base) == _candidate_signature(list(base))
