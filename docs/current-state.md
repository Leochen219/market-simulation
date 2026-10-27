# Research state: 27 October 2026

The objective is to model conditional human behavior given market state and a participant's history. Engineering execution, profitability, and behavioral fidelity are distinct questions. There is no independently validated digital human or general production default.

In the completed conditional-distribution study, the prespecified candidate's eight-dimensional energy score was 0.6840 versus 0.2734 for a state-retrieval reference; lower is better. The retrieval reference used additional development-pool information, so this is not a controlled comparison changing only the model. These observations are from one market and do not establish performance across markets.

A fixed-direction one-order diagnostic compared three- and ten-round histories for 18 development participants. Quantity W1 fell from 21.75 to 18.75 for BUY and 32.75 to 25.25 for SELL. BUY price W1 was 198.40 versus 210.50; SELL price W1 was 323.59 versus 252.34 on 15 of 16 paired successful paths. Missing output was not imputed as zero; the long-history SELL price W1 bounds were 202.41–264.69. W1 measures distributional distance, not per-person accuracy. The fixed direction and one-order scope exclude direction, count and model STOP prediction.

The price-grid spacing was 2.5 across 2.5–997.5; projecting original integer prices onto the grid can move a value at most 1.5. In the long-history BUY group, observed and model mean prices were 19.75 and 229.75 respectively, with W1 210.50. Thus rounding alone does not explain the gap. Sequential thresholding and renormalization can amplify low-probability conditional peaks; saved probabilities also contain high-price mass. These observations do not isolate a single cause or establish model calibration.

A client-side empty-support stop rule was evaluated only as an offline replay of 72 existing paths: 71 completed unchanged and one stopped; 22 successful high-price outputs remained. This does not show improved human prediction, and is separate from model-initiated STOP. No new 20-round Jev market was run for these diagnostics. Participant-level records and raw inputs are not redistributed.
