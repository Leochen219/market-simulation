# SELL economic semantics: paired development comparison

## Authorized question and scope

The user authorizes a small real-Jev comparison clarifying auction rules, retail value and SELL limits, using the same people/history, top-three plus raw 3% cutoff, and ten-way concurrency. Analyze people with and without personal SELL history separately. Do not run a 20-round market. This design continues the earlier neutral-definition proposal already accepted by the user; routine implementation choices below are root rulings.

Use the eight supplied-SELL people from the frozen `jev-flat-grid5-study-v1-2026-10-27` baseline (plan SHA256 `8891b9242ff05f4df240ceac454a9aa7c3cccfe47ef581d9f458febbbe8aa281`). Target remains formal11/game16. Short3 uses formal8–10; long10 uses formal1–10. Keep each baseline path's direction, state, history, quantity and repeat. Two repeats yield32 price paths; run both ORIGINAL and RULES contemporaneously, for64 actual price requests. Do not rerun quantity or add BUY cases, tools, transaction feedback, oracle conditions or historical examples from other people.

## One input intervention

ORIGINAL must serialize byte-for-byte identically to the corresponding old flat5 request. RULES is constructed from that same request/state, adding only `public_instructions.economic_definitions`. Questions, observer instructions, candidates/order, numbers, history, partial-order ledger and all existing fields are identical. No arm name enters the model-visible input. Full199 complete dollar values5,10,...995 remain legal.

Freeze the following English text:

> own_p is revenue per unit of this participant's own demand D served, not an order limit or market price. Other participants have different private p. The wholesale cost of Q was incurred before D was revealed. Units neither used to serve own demand nor sold expire at round end with no salvage value. Orders are blind limits: a BUY limit is the maximum acceptable unit price; a SELL limit is the minimum. Bids rank highest first and asks lowest first. Fills may be partial or zero. All fills in a market trade at one uniform price: the midpoint of the last matched bid and ask. An order's fill price can differ from its limit; a limit can affect the clearing price when marginal. Historical order prices are submitted limits, not realized prices. These definitions describe the rules, not a recommended action.

This is a bundle-of-definitions intervention. It does not identify which sentence is causal and does not instruct profit maximization, a low ask, a target action or a computed current payoff. The matched quantity is an old model choice, not the human target quantity. Human orders remain scoring-only; original dollar1 can be outside the new grid.

## Sampling, schedule and failure policy

Keep `jev-1.14.0`, pinned SDK, full candidate order and the existing raw cutoff3 decoder. Keep original seed/person/target/repeat/price draw coordinates, omitting arm and history window from the draw exactly as before. Physical IDs retain arm identity. This gives common draws within each pair; it does not force identical provider responses.

Deterministically shuffle32 arm requests within each repeat, balancing which arm precedes its mate; save the full schedule and its seed. Repeats are separate time blocks; batches have at most10 requests and600 seconds. Maintain ten-inflight,40RPS and50,000 reserved bytes/second. No retries, threshold relaxation, partial completion scored as STOP, or substitution of historical responses. Valid empty support is client numeric abstention; transport/invalid probabilities remain technical failure and halt new dispatch.

Separate ledger hard limits:64 requests and250,000,000 nanodollars (USD0.25). Reserve65,536 input tokens at42 nanodollars/token before each dispatch, including uncertain failed attempts; maximal64-request reservation is USD0.176160768. No long context/compression. Every fully serialized body must be at most12,000 bytes; check all previews and again immediately before dispatch. Do not truncate history. A one-time start marker prevents automatic reruns. Freeze code, neutral text, prompts, candidate order, body hashes, baseline/source hashes, labels, draws, schedule and fee contract before live calls.

## Evaluation fixed before execution

Primary: per-window change RULES minus ORIGINAL in **person-balanced pooled price W1** to raw human same-side submitted prices, on identical successful arm-pair paths. Negative is closer. This is a distribution objective, not mean individual prediction error. Persons are equal weight, with equal human order weights within person and equal successful repeat weights within person. Report mean/median/P95 and CDFs with the same weighting.

Report ALL, HAS_SELL_HISTORY and NO_SELL_HISTORY groups per window. Compute history membership from the model-visible raw historical orders only, never target orders/profits. Existing data has seven versus one; the one-person group is a descriptive case, not evidence about an interaction or a population. Save full private per-person differences for audit, with only safe aggregates public.

Always report valid quote counts/planned requests, client abstention, technical failures, prices>=65,>=700,>250 and above own retail p. These are descriptive events, not automatically economic mistakes. No quote is neither zero nor human STOP. Show individual raw-score totals and normalized pre-cutoff mass<=20 and>=65, retained-distribution mass and actual samples separately; none establishes calibration. A decline in mean alone is not success. Include raw-human versus common5-dollar human projection sensitivity and price-grid/conditioned-quantity caveats. Original history and previous flat5 outcomes remain frozen; the contemporaneous ORIGINAL is the main control.

Any technical-incomplete run remains incomplete; preserve denominators and evidence, refuse a complete-comparison report, and do not retry. A complete run with valid abstentions reports coverage and common-support limitations explicitly. Two repeats, eight people in one previously used development market do not provide independent confirmation or a causal error decomposition. A null result does not establish intrinsic Jev inability.

## Offline acceptance and delivery

Prove original-body equality, exact one-field intervention, no input mutation/target leakage, full price grid and boundary legality, history-based stratification, paired reproducible draws/schedule, budget reservation and one-shot start, valid abstention versus technical failure, ten distinct mock requests overlapping, and strict replay/tamper detection for both successful and failed wire evidence. Root integrates and independently reviews before the one live run.

Deliver frozen generic prompt, private full inputs/responses/selection replay, safe JSON/Chinese comparison and CDFs, tests and local commands. Full subgroup metrics/CDFs stay private: the no-SELL group has one person, and publishing ALL plus its seven-person complement can reconstruct that person's distribution. Public JSON/figures and Notion contain only ALL-group numeric distributions, group counts and appropriately qualitative subgroup findings; the local user report contains both strata. Update the existing Notion homepage only; publish safe code/aggregates to the existing GitHub PR without participant records or credentials. Retain all old versions. No new Notion page or market run.
