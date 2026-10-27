# Conditional behavior study: pre-dispatch cost gate

Checked 2026-10-26. This record concerns `jev_conditional_behavior_v1`, not historical experiments. No paid calls were needed for this check.

## Jev

Use the pinned `jev-1.14.0` standard endpoint, no long-context header, no images, no automatic retries, and at most 12,000 bytes for the complete UTF-8 request body. The [model documentation](https://docs.typesafe.ai/models) gives a 64k aggregate request context, state ingestion once, and USD 0.042 per million input tokens with free outputs. The [API response definition](https://docs.typesafe.ai/api/system-one) identifies `usage.input_tokens` as the billable input count for the request.

The root and Astra architect accept the ordinary interpretation of that documented contract: billable input is governed by the aggregate request context limit. Reserve 65,536 input tokens per physical HTTP attempt, including a packet of several questions. This is USD 0.002752512 per attempt, using the more conservative binary interpretation of 64k. This bound is not a byte/token estimate and is not inferred from historical average usage.

The guarantee is conditional on the published provider contract: no separately billed hidden input, rejection surcharge, or repeated-state billing beyond that limit. Internal system processing is treated as included in published pricing. These assumptions are not independently confirmed provider guarantees. Do not deliberately submit oversized requests. Keep the entire reservation on rejection, timeout, missing usage, cancellation, or unknown billing; do not presume failures are free.

Before sending each request, atomically persist and check:

`settled USD + unresolved/in-flight reservations + next reservation <= 20 USD`.

Only validated authoritative usage may release part of a reservation. A usage count above 65,536, malformed usage, or an unexpected model must halt further dispatch and retain evidence. Record the actual outgoing header policy, body hash, request identifier, timestamps, model, usage, and settlement. Persist before network send so a crash cannot erase a possible charge. A new output directory must not reset a study's shared spending cap.

The full maximum request count reserved without settlements would exceed USD 20. Therefore completion depends on real usage releasing unused reservations; affordability must be previewed, and a budget stop must be reported as incomplete rather than analyzed as a complete study. Do not silently increase the cap or spend the budget on diagnostics before funding the protected human comparison.

## Opus

The existing installed collaboration helper passes `--max-budget-usd`, a timeout, high effort, and no tools/MCP. It does not establish a hard total cost bound for a complete invocation. The [official CLI reference](https://code.claude.com/docs/en/cli-reference) explicitly warns that spending can exceed this cap; the [agent-loop documentation](https://code.claude.com/docs/en/agent-sdk/agent-loop#budget-headroom) explains that the check occurs after responses arrive.

The user subsequently clarified on 2026-10-26 that this existing Claude account uses subscription quota and explicitly removed the USD 10 ceiling for Opus. The 48 predeclared synthetic reference consultations may therefore use the existing helper. Keep its bounded per-invocation timeout, request-size limit, no tools/MCP, no automatic retry, and fixed model/receipt checks. Any helper budget flag is an operational soft cap, not a claimed hard financial guarantee. Record reported list-cost estimates for provenance; do not describe them as subscription debits. This change does not lift Jev's USD 20 cap, authorize additional conditions, enable extra usage, purchase credits, or change accounts. No helper modification is needed.

The Opus count refers to helper/CLI consultations, not a proven count of underlying provider HTTP requests. The experiment driver never repeats a failed consultation. The installed helper does not expose all native CLI internal retry controls; do not claim that 48 consultations prove exactly 48 underlying model requests. Jev physical request counts, by contrast, are controlled and recorded directly.

## Consultation provenance

The cost reasoning received a read-only local Astra architect review. It was not an Opus consultation. The project's user-paused quota routing remains paused; this document does not restore it.
