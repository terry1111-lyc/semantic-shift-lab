# Data card — synthetic pilot v0.1

**Source:** original deterministic English templates authored for this project. No downloaded benchmark items or human participant data are included.

**License:** MIT for the original code/templates and shipped generated fixtures. Any later external dataset retains its own license.

**Intended use:** software validation and initial controlled arithmetic-language experiments. Not a validated benchmark, broad reasoning score, or deployment qualification.

## Composition

Three broad families: card transfers, book-count comparisons, and grouping balls in boxes. Three templates per family; four variants per group. Default `per_template=20` yields 180 groups / 720 records; 60 groups per split. Names, vocabulary and task scope are narrow.

Template index 0 is train, 1 is development and 2 is test. This partition is explicit and reproducible but not randomized. Split difficulty may differ. Family and arithmetic substructures overlap; avoid calling it a strict compositional-generalization benchmark. Performance across splits is not directly comparable without accounting for their content.

## Fields

| Field | Meaning |
|---|---|
| id | Unique record ID |
| group_id | All four variants share this |
| family | Broad task category |
| template_id | Authored template identifier |
| split | train/dev/test |
| variant | original/paraphrase/reorder/edit |
| prompt | English question only |
| program | Nested arithmetic AST, never passed to a real inference backend |
| answer | Exact integer answer from AST |
| review_status | Initially unreviewed; approved only after actual review |
| schema_version | 1 |

## Validation and remaining risk

Checks cover duplicate IDs, missing variants, invalid ASTs, label mismatch, answer-changing edit constraints, group metadata, template leakage and exact normalized-prompt leakage across splits. The validator does not verify English semantics or guarantee absence of near-duplicates. Review the templates and generated instances manually.

Changing `seed` changes numeric instances, not structural diversity. Generating many examples from nine templates does not make a large independent benchmark. Only three held-out templates means bootstrap intervals remain exploratory.

## Versioning and exclusions

Freeze a reviewed dataset and record its SHA256 before inference. Keep a review ledger and exclusion counts. Do not select items based on model failure. If a variant is invalid, resolve it before evaluation or exclude the entire group. Preserve the original and corrected dataset versions. No guarantee is made that a pretraining corpus lacked similar arithmetic questions.
