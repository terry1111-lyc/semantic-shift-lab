# Failure cases / 失败案例

Observed examples from the completed study are documented separately in [actual-model failures](../research/local-pilot-fp32/OBSERVED_FAILURES.md).

**All examples on this page are designed illustrations. None is an observed LLM finding.** The fixture's errors are deliberately injected by code that can see labels.

## 1. Ignoring a relevant edit

Original: 18 cards, receive 7, give away 5 → **20**.
Edited: 18 cards, give away 7, give away 5 → **6**.
Hypothetical failure: answer 20 to both.

Follow-up: change names/numbers and direction wording independently; check arithmetic with the correct equation. Do not conclude from one case that a model memorized a template.

## 2. Relation reversal under equivalent wording

Original: Mia has 8 books; Noah has 3 more → Noah has **11**.
Rewrite: Mia has 8 books, which is 3 fewer than Noah has → Noah has **11**.
Hypothetical failure: answer 5 to the rewrite.

Follow-up: separate comparative-language comprehension from arithmetic. Record correct and incorrect controls.

## 3. Generator semantic drift

Original: three boxes, **six balls in each** → **18**.
Bad rewrite: three boxes containing **six balls in total** → **6**.

The answers should differ. Treating this as an invariant pair is a data error, not a model error. Quarantine the group, correct the generator and version the dataset before evaluating again.

## 4. Parser false negative

Gold: 20. Output: “He has twenty cards.”

The current strict integer parser rejects this. It reports a format/parser failure, preserving raw text. An independent audit may adjudicate the response as semantically correct. Publish both strict and adjudicated scores with the policy; do not silently relax parsing for one model.

## 5. A method fails the control experiment

Hypothetical: E beats C initially, but the gap vanishes when independent original-data quantity is matched. Interpretation: current evidence supports a data-coverage explanation, not an extra effect of the proposed mixture. Do not report the original gap as proof of a novel training method.

## 6. Generalization fails

Hypothetical: performance rises on seen templates but not on independently authored structures or external tasks. Interpretation: the improvement is restricted to covered variations. Preserve the failed test and narrow the claim.

## 7. The hypothesis fails

Hypothetical: rewrite augmentation improves both metrics across seeds; H2's proposed trade-off is absent. Report the null/contrary result. Seek boundary conditions only in separately labeled exploratory work.

## Real-case record template

```json
{
  "case_id": "fill-after-an-actual-run",
  "run_id": "immutable-run-reference",
  "group_id": "dataset-group-id",
  "raw_outputs": {},
  "strict_scores": {},
  "human_category": "pending",
  "data_semantics_checked": false,
  "alternative_explanations": [],
  "follow_up_experiment": "",
  "is_observed_model_result": false
}
```

Only set `is_observed_model_result` to true when backed by a real model run. A fine-grained error label requires review; output wording alone cannot reveal a model's internal reasoning.
