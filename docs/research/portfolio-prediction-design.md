# Autonomous human-portfolio prediction pilot

Status: root design after Astra code inspection and a verified, tool-disabled Opus architect consultation. It implements the active human-decision goal; no market run or publication is authorized here.

## Scientific contract

Predict complete submitted order portfolios conditional on each human's actual pre-trading `p,w,Q,D`. This isolates trading from procurement errors. It is not a joint autonomous procurement/market rollout. First- and third-person prompts have the same participant-behavior forecasting objective and differ only in their perspective sentence. Preserve all earlier profiles and prompt snapshots.

Select three new GA-10 participants using seeded ID hashes, excluding the original procurement pilot's three. Fix eligibility to all formal rounds 1–10 present and valid; fail rather than replace based on target behavior. Use their last three order episodes from a corpus limited to formal rounds 1–8 for both held-out rounds 9 and 10. No selection or tuning between these rounds; call both held-out, not development/confirmation.

The [paper](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03771), §3.1, §4.1, Table 2 and Appendix B, supports: demand known before trading, up to five orders, quantity and integer-dollar limits 1..999, gross submitted sells <= initial Q, blind one-shot clearing, and GA-10's 35 participants. Formal equal-price crossing uses >=. Include the human-visible no-trade pretrade projection, calculated from already known p,w,Q,D; it is not a forecast of fills or a recommendation. Same-round inventory expires.

Workbook rounds are the fifteen analyzed formal rounds, after five warmups. Keep source-selection/scoring indices formal 9 and 10, but display game rounds 14 and 15 with horizon 20; display frozen historical formal rounds 6–8 as game rounds 11–13. Do not accidentally label round 10 as the last game round. Trace request metadata must use the same declared game clock.

Do not manufacture a cash constraint, prohibit mixed/duplicate/crossed quotes, sort generation, or prune strange legal prices. Platform cash enforcement, tie allocation and self-matching remain unresolved interpretations, explicitly recorded. Public market size 35 is established here from paper evidence plus the verified roster; it is distinct from the API concurrency cap 10. The earlier pilot's unknown-N input remains unchanged in its separate diagnostic.

Limitations: frozen history omits round-9 feedback for round 10 and omits explicit historical fill/profit fields; it supplies three prior decision examples rather than exactly recreating the two-round UI. No spreadsheet/arithmetic calculator is available in this baseline, although humans had support tools. The old guaranteed-full-fill payoff tool is inappropriate as a substitute. No assertion that Q or wholesale cost mechanically carries over from previous fills: the paper has per-round expiry and independent new demand/retail draws.

## Generation and acceptance state

Twelve independent portfolios: 3 people × 2 held-out rounds × 2 perspectives, one repetition. Each agent uses only its own generated trace:

1. At slots 0..4 choose STOP/BUY/SELL; remove SELL only when remaining gross sell capacity is zero.
2. BUY quantity 1..999; SELL quantity 1..remaining gross capacity; price 1..999.
3. Decode each number with own hundreds/tens/ones prefixes and legal masks; singleton masks are recorded as forced constraints without a model call.
4. Append a completed order only after all its components validate. STOP or five completed orders finishes. A tied, failed, invalid or truncated prefix marks the whole portfolio incomplete, never a shorter successful portfolio.

Use the existing `PaperOrder`, `validate_portfolio`, `prefix_mask`, economic candidate labels and reply validation. Do not reuse an instruction that says maximize profit. No other trader's current order, held-out target component, clearing price or outcome enters generation.

## Freeze, guard, evidence and budget

Pre-enumerating all order histories is infeasible. Freeze static target facts/history, prompt text, schema, transition/candidate/mask/decoder versions, model and pinned SDK version, source/policy hashes, selection seed, scorer/baseline/protocol code, and the twelve planned sessions before credentials.

Preview writes separate private input/manifest and scoring-target files. Manifest binds the scoring file hash; the inference process reads only frozen inputs and manifest. Offline scoring reads targets after generation finishes. Require execute to load an explicit reviewed preview, verify all hashes, and reject a changed protocol/code/base. Use a fresh output directory with no resume or automatic retries.

For every step reconstruct the expected request independently from immutable base plus accepted trace. Register an expected semantic wire for its unique request ID before backend dispatch. Require exact state/instruction/criteria/model equality, never a union of self-registered outgoing hashes. Preserve the existing ContextVar binding and fail closed on absent/duplicate/consumed IDs.

Pin `compression=False`. A dry SDK client with dummy key and MockTransport serializes the independently expected request, without network or model work. Register that exact body hash and method/path. The live transport validates the actual uncompressed body, byte bound and request context against the one-shot entry before sending. Never retain Authorization/header values. Test actual SDK serialization and ten concurrent contexts. This retains SDK parsing without assuming its serialization equals a handwritten JSON encoder.

Bind the full reviewed destination tuple (scheme, host, port, path and query), explicitly overriding environment base-URL changes; disable redirects. The dry and live serialization must have identical body-affecting options. Missing context fails closed. Do not normalize away unexpected byte differences or accept another legitimate session's body.

New experiment limits, separate from procurement: at most 35 decisions per portfolio / **420 actual requests**, **7,500 uncompressed HTTP-body bytes per request**, and **3.2 million total estimated bytes**. Reserve 7,500 bytes per model request, conservatively totaling <=3.15 million. One shared limiter: 40 RPS, 50,000 reserved bytes/sec, 10 in flight, SDK retries 0. Keep actual body-byte totals separate from these reservations and from provider billing tokens.

Preflight a maximal-width state/criteria envelope for each target and view, including five worst-width pending order records as a conservative superset, all numeric optional fields and ten longest candidate labels. Prove the 7,500 bound using closed bounded schemas, and reject oversize bodies again at transport. Do not pad real state values or truncate history to fit; that would change the interface. Runtime guard consumes one-shot entries and persists/fsyncs a request-attempt ledger before HTTP send. A crash retains charges; execution never resumes or overwrites the old output. New experiment IDs have independent, explicit budgets.

Hash-chain private trace events. Record expected/actual wire and body hashes, raw request/response bodies without headers, accepted choices, forced constraints, failures and HTTP intervals. Offline replay reconstructs all requests, masks, choices and final portfolios without API calls; reject tampering or truncation. Truth mutation must not affect any generated request.

A valid terminal run record binds the frozen manifest hash, all twelve final statuses, every final trace-chain head and ledger count. A chain without its terminal record is incomplete, because internal chain validity alone cannot detect suffix deletion. A crash may leave a charged attempt without a response; report it as unresolved, never automatically retry it. Inference retains the manifest's declared scoring-file hash but never opens that file; only offline scoring verifies it.

## Scoring

For integer x=1..999, define B(x) as total BUY quantity with limit >=x, S(x) as total SELL quantity with limit <=x. Primary error is `(sum |B_pred-B_human| + sum |S_pred-S_human|)/(2*999)`, measured in quantity units per price point and side. Secondary window 1..100 covers the paper retail-value domain; report it separately without restricting legal prices.

Main score uses every planned person-round target in each arm. A missing portfolio gets the conservative upper bound `U = sum_x(max(B_h,Bmax-B_h) + max(S_h,Smax-S_h))/(2*999)`, with Bmax=4995, Smax=Q. It cannot outperform any legal complete portfolio. Also report completed-case error and coverage, failure categories, paired-view differences on all six targets, and both-completed comparisons. No significance or population-equivalence claims with three people.

Secondary: gross buy/sell totals, empty agreement, side-use class, order-count error, exact multiset equality, and merged same-side/same-price quote equality. Curves ignore slot permutation and order splitting; merged equality does not prove auction equivalence under tie rationing. Keep BUY and SELL curves separate so opposing orders cannot cancel in the score.

Training-only baselines: last round-8 portfolio, pointwise personal median curves, and empty portfolio, evaluated on all the same targets. Report historical baseline feasibility under current Q, without clipping it. Median curves are statistical references and may not be executable five-order portfolios. Realized fills/profit are not decision-similarity endpoints here.

## Required tests and report

Immediate STOP; five orders; mixed sides; duplicates; 999 quantity/price; price 1; Q=0; exhausted sell cap; buys never replenish it; leading zeros; singleton constraints; ties and failed partial orders; provider errors; oversize and duplicate/missing-context sends; ten real-SDK/mock concurrent requests; truth/outcome mutation invariance; changed own trace changes later requests; paired prompt/state parity; replay tampering; curve permutation/splitting invariance; U bounds extreme legal errors; baseline training isolation; closed schema/no ID leakage; scoring-target unreadability during inference.

Before live: full tests, independent cross-provider review, frozen preview comparison and source-domain audit. After live: request/body/ledger/trace accounting, HTTP overlap, deterministic replay, safe aggregate report and prompt snapshots. Keep all target states, labels, traces and per-target residuals ignored and private. No 20 rounds, banks, credit, publication, or claim of a full-market reproduction.

## Advisory provenance and root decisions

Astra inspected the existing code and recommended the conditional order-only pilot and transition/replay boundary. Opus (`claude-opus-5-5`, receipt complete, no tools/MCP/API key; list/API cost $0.311844) reviewed a minimized design summary, not raw records or code. Root adopted exact-HTTP-body validation, persistent no-resume accounting, conservative missing-portfolio scoring and tool-free labeling. Root rejected unnecessary real-state zero padding and the suggestion that next-round w/Q mechanically carry over from fills. Current source p is retail value, not clearing price. Published instructions establish numeric ranges; original platform cash/self-match details are still not claimed as verified.
