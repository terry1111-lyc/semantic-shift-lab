# Observed measurement failure — real development outputs

This diagnostic used 12 development examples (three groups), before the held-out test study. It is not a generalization estimate. Review was performed by the assistant, not an independent human annotator.

With the initial brief-calculation prompt and 96-token cap, the strict parser scored all 12 as incorrect. Eight responses were not truncated; all eight visibly stated the correct mathematical answer in LaTeX or Markdown, while failing the required `FINAL: <integer>` syntax. Four responses hit the token cap.

Example: Lin has 98 cards, gives away 4 and then receives 11. Gold = 105. The model computed `98 - 4 = 94` and `94 + 11 = 105`, emitted `\boxed{105}`, then literally wrote `FINAL: <integer>`. Strict score = incorrect despite the correct displayed result.

Under the development-only answer-only prompt, the same edit produced `40`. Thus changing output instructions can affect substantive answers, not only formatting. The main answer-only pilot must be interpreted as a specific protocol, not a measure of all reasoning capabilities.

Raw outputs and per-item audit decisions are preserved. No primary test score was retroactively changed.
