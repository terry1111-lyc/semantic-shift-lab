# Actual-model pilot: quantitative results

**These are actual Qwen3-0.6B outputs, not the repository demo fixtures.**

| Run | Original | Invariant all-correct | Edit-pair correct | All-four correct | Format failures | Token cap |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0% | 0.0% | 0.0% | 0.0% | 74.2% | 34.2% |
| B-seed17 | 26.7% | 3.3% | 18.3% | 3.3% | 0.4% | 0.4% |
| C-seed17 | 20.0% | 1.7% | 5.0% | 0.0% | 0.0% | 0.0% |
| E-seed17 | 28.3% | 3.3% | 1.7% | 0.0% | 0.0% | 0.0% |
| B-seed29 | 26.7% | 0.0% | 16.7% | 0.0% | 1.2% | 1.2% |
| C-seed29 | 16.7% | 0.0% | 3.3% | 0.0% | 0.4% | 0.0% |
| E-seed29 | 31.7% | 1.7% | 20.0% | 1.7% | 1.2% | 1.2% |

## Primary paired comparison

E minus C on the same test groups, separately for each seed. Intervals resample only three template clusters and are exploratory.
- E-seed17 minus C-seed17: +0.0 percentage points; interval [+0.0, +0.0].
- E-seed29 minus C-seed29: +1.7 percentage points; interval [+0.0, +5.0].

## Interpretation limits

Three template clusters and two training seeds; exploratory only. Format and truncation confounds must be considered. No diversity-matched baseline.
Answer-only generation is capped at 16 new tokens. A format failure or truncated answer cannot by itself establish a reasoning failure. The original longer development run is preserved separately.
Conditional accuracy among parseable outputs is a diagnostic with unequal denominators, not a fair replacement for the primary metric.
All runs share the same model snapshot, test set and decoding. Training uses a small matched total-token budget but different unique-example distributions. The study does not test RL or longer reasoning.
