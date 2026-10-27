# Behavioral memory: pre-dispatch cost contract

Checked2026-10-26 for `jev_behavioral_memory_v1`. User authorized this study with a NEW USD12 cap; the older conditional-study USD20 ledger is separate.

The pinned standard `jev-1.14.0` endpoint has 64k aggregate request context; state is ingested once. Published price is USD0.042 per million input tokens, outputs free. The API describes `usage.input_tokens` as billed input. Sources: https://docs.typesafe.ai/models and https://docs.typesafe.ai/api/system-one . No long-context or image header is allowed.

Use the same conservative contractual interpretation as the prior reviewed cost gate: reserve65,536 input tokens for EACH physical attempt, USD0.002752512 (2,752,512 nanodollars). The request body cap of12,000UTF-8bytes is a separate guard, not a tokenizer estimate. At3,888 attempts, reservation totalsUSD10.701766656, withinUSD12.

This bound depends on the published contract including all billable input within the aggregate context limit, with no separate hidden-input/rejection surcharge. It is not an independently guaranteed provider upper bound. Before each send persist an atomic SQLite reservation: settled+inflight+unknown+next <=12,000,000,000nanodollars, and attempts<=3888. Never release reservations for uncertain failures/cancellations/timeouts/missingusage. Only authoritative validusage for the pinned model can settle. Usage>65536 or model mismatch halts further dispatch and keeps evidence.

Use fixed `artifacts/participant-data/jev-behavioral-memory-v1/cost-ledger.sqlite` across run/output directories; each preview/attempt must bind to it. Refuse repeated live execution of the same frozen study. Freeze source, policy, code, complete request preview, prompt, seed/schedule and scoring before reading credentials or dispatching. Root attestation binds their hashes plus this file and the absolute ledger path. All raw requests/answers and participant records remain local in the ignored participant-data tree.

10inflight,40RPS,50,000reservedbytes/second,12,000actualbodybytes,600seconds per batch of at most10trajectories. Retries/resumption/automatic newrun are disabled. Per-request evidence and global budget reservations are durable before send. A failed comparison stays incomplete; budget cannot be increased automatically.

Opus implementation reviews use the already authorized membership consultation channel. Its list-cost estimate is not a subscription debit. The user's quota-routing pause remains in force; authentication preflight is used only for an actual consultation.
