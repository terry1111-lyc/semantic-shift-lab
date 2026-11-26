# Float32 engineering rerun — protocol v3

This rerun follows the original registered pilot. All model weights, data, prompts, budgets, seeds, and primary comparisons remain fixed. CPU precision is now explicitly float32, and inter-op threads are limited to one. Batch size is eight for all arms.

The original bfloat16 baseline was inspected before this rerun; this is not a newly blinded confirmatory test. The precision decision followed a bounded development-only training benchmark (three steps), not E/C test performance. No E/C training results had been produced.

The original attempt and measurement failures are preserved under ../local-pilot/. The primary comparison remains E minus C on the same groups for seeds 17 and 29. Three test templates, two seeds, small budgets and unequal variant counts only support exploratory conclusions.

A 15-minute watchdog bounds each run process; any timeout must be reported as incomplete rather than scored.
