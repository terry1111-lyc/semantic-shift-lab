# Actual-model pilot: quantitative results

**These are actual Qwen3-0.6B outputs, not the repository demo fixtures.**

| Run | Original | Invariant all-correct | Edit-pair correct | All-four correct | Format failures | Token cap |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0% | 0.0% | 0.0% | 0.0% | 71.7% | 31.7% |

## Primary paired comparison

E minus C on the same test groups, separately for each seed. Intervals resample only three template clusters and are exploratory.

## Interpretation limits

Three template clusters and two training seeds; exploratory only. Format and truncation confounds must be considered. No diversity-matched baseline.
Answer-only generation is capped at 16 new tokens. A format failure or truncated answer cannot by itself establish a reasoning failure. The original longer development run is preserved separately.
Conditional accuracy among parseable outputs is a diagnostic with unequal denominators, not a fair replacement for the primary metric.
All runs share the same model snapshot, test set and decoding. Training uses a small matched total-token budget but different unique-example distributions. The study does not test RL or longer reasoning.
