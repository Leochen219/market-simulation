# SELL semantics study operations

This isolated study matches 32 frozen SELL price paths with two contemporaneous arms. The ORIGINAL request is verified against the parent frozen request hash. RULES adds only the frozen economic definitions. Person, history window, quantity, candidate order and deterministic sampling coordinate are unchanged. No arm label enters the wire input.

Preview with `python tools/probes/jev_sell_semantics.py preview`; after independent verification, use `freeze`, then `run` once. The start marker is exclusive and is created after plan validation, credential availability and ledger initialization, before any request. The runner saves request, response, timing, failure and cost evidence privately. Use `replay` to verify it and `report` to compute the comparison. Do not retry partial runs. The 64-attempt cap reserves 65,536 input tokens at 42 nanodollars each, with a 250,000,000 nanodollar ceiling. Requests use the original client limits and no retries.

The paired study cannot isolate individual sentences in the definition bundle. Abstentions are not zero prices; technical failures leave the comparison incomplete. The data comprises eight development participants with two repeats and is not independent confirmation.
