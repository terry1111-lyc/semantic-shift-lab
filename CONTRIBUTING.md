# Contributing

Start with `python -m unittest discover -s tests -v`. Core functionality uses only the Python standard library.

For data changes, add hand-checked semantic cases, verify the arithmetic program, and ensure group/template split integrity. Keep a variant's meaning and target answer explicit. Tests must cover semantic changes, not just call the same generator twice.

For scoring changes, include ambiguous-output and missing-prediction regressions. Version the parser and keep original raw outputs; never silently change previously reported scores.

For model/training changes, run the opt-in tiny-model integration test. Do not require remote weights or credentials for core CI. Never commit tokens or cached model weights.

Describe what was changed, why it matters, how it was verified, and what remains uncertain. Separate demonstrated observations from proposed explanations. Related-work claims require sources; no fabricated results or author credentials.
