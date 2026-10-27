# Publication verification — 27 October 2026

The publication snapshot was verified on Windows with CPython 3.11 and the existing pinned dependency runtime, importing source from this exported repository. A completely blank environment installation was not independently tested.

- Full offline suite: **576 passed, 13 skipped**, plus **12 subtests passed** (589 collected tests).
- Ten specifically marked tests require the undistributed authorized participant source, policy and registries. They skip when those prerequisites are absent; invalid present data still fails validation. Three other tests skip because their historical private evidence or reviewed pilot artifacts are not distributed.
- Synthetic random-legal paper profile: ten agents, twenty rounds, five warmup and fifteen formal rounds, completed. Strict replay and summary completed without model inference.
- Python wheel build completed: `capclaims-0.1.0-py3-none-any.whl`.
- The staged publication tree was checked for forbidden private-data paths and literal credentials, including a local-only comparison against the configured Jev key. No credential values were written to this repository.

No live Jev calls, model downloads or participant-data experiments were performed for this publication validation. Generated artifacts and local environments are not included.

The completed suite used `python -m pytest tests -q -p no:faulthandler`. An earlier run with an added 90-second faulthandler timeout ended in a Windows native access violation while dumping a stack; that interrupted run was not counted as passing. One synthetic end-to-end test normally took about 107 seconds in the completed run.

The repository is a clean publication snapshot. Original private run logs and frozen research versions remain in the local research workspace; replaying those old studies requires their exact authorized evidence and matching source hashes. Research document redactions and release-only test availability guards are not retroactive changes to those results.
