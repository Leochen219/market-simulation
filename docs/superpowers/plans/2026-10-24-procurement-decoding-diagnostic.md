# Next step: distinguish greedy decoding from the elicited quantity distribution

Status: implemented after the completed procurement pilot; not yet executed. Await independent review before any real provider call. Overall human-decision goal remains active. This is an exploratory mechanism diagnostic on development inputs, not a new confirmation of human fidelity.

## Reason

The frozen pilot generated no completed quantity below 100, including when some visible history was below 100. First-digit probabilities did assign nonzero mass to 0..99. Demand-mean anchoring, round-number preference, history effects, and the digit interface remain competing explanations. Opus recommended enumerating the existing prefix tree before changing prompt/state wording.

## Frozen scope and budget

- Choose the first two participant aliases in the original seeded selection, development round 9, `first_own`, repetition 1. Selection uses only alias order and round, not error size or human answers.
- Reuse the exact current state, instruction profile, candidates, and all legal prefixes from the reviewed original plan. Never supply human digits to construct a path.
- Enumerate all 47 nonterminal legal prefixes for Q=0..400. Two prefixes, `4` and `40`, have only one legal continuation; assign that continuation probability 1 as a deterministic legality constraint without a model call.
- Query every remaining prefix afresh: 45 per input, at most **90 additional requests**. Combined with the completed pilot's 377, this remains at most 467 within the earlier 480-request planning ceiling. No retries. Preserve 10-way concurrency and the same rate/byte limits.
- Preflight exact byte costs and require combined completed-pilot plus diagnostic estimated bytes <=3.2 million. Fail before loading credentials if either bound fails. No hidden reuse of confirmation inputs.
- Freeze source, original plan, current code, protocol, prefix wires and selected inputs before provider construction. Keep participant states/labels private under ignored artifacts. Record singleton paths separately so they are not described as model predictions.

## Analysis

For each integer q with digits h,t,u, compute the induced tree probability `P(h) * P(t|h) * P(u|ht)`. Accept backend conditional sums only within its 1e-6 validation tolerance, normalize each accepted row before multiplying, and preserve raw row totals privately. Verify nonnegative finite probabilities and post-normalization total mass 1 over all 401 quantities. This is the distribution induced by the conditional elicitation protocol; it is not a verified native flat-choice distribution or calibrated human behavior probability.

From the same freshly evaluated tree compare:

1. Greedy per-digit quantity, with top ties unresolved.
2. Global modal quantity or full tied mode set.
3. Integer median (smallest q reaching cumulative mass >=0.5), appropriate for MAE under this induced distribution.
4. Mass below 100, expected quantity, and paired differences between decoders.

Report original saved greedy quantity separately from fresh-tree greedy quantity; provider variation must not be mistaken for a decoder effect. Score human quantities only after the tree is complete, locally. With two development inputs, report mechanism diagnostics and no significance, population fit, or best-prompt claim. Do not tune or choose an interface on confirmation rounds 12–15.

## Engineering acceptance

- Unit tests cover all 401 leaves including 400, normalized mass, a constructed example where greedy differs from joint mode, ties, and singleton constraints.
- Mock end-to-end test sends only legal frozen wires, <=90 calls, one shared limiter, and no credentials/network.
- `--execute` requires `--preview-plan` pointing at `frozen-diagnostic-plan.json` from a reviewed preview. Rebuild and compare the full plan, protocol text/hash, code hashes, parent checksum, and selected wires before loading credentials; a changed or missing preview blocks execution.
- Independent review before real calls; compare execution plan to frozen preview and verify actual HTTP request counts and parallel intervals afterward. A successful live run requires one recorded HTTP interval per adapter request and no more than ten overlapping intervals.
- Preserve original v3, procurement prompts, pilot results and raw evidence. Any subsequent representation change needs another version and fresh held-out participants.

After this diagnostic, specify a matched historical-feedback arm using only facts genuinely visible before procurement. Then evaluate on new participants before claiming any improvement. Full autonomous portfolios remain a separate next acceptance stage; do not infer them from procurement results.
