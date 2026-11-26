# Training interruption record

The baseline inference run A completed (240 inputs). The B-seed17 training attempt logged step 10 and 635 processed tokens, then had no new progress log for approximately 4.5 hours of wall-clock time. The machine may have slept; this is not a measured 4.5 hours of active CPU computation. No completed checkpoint was written.

On resuming the task on 2026-11-26 (Australia/Sydney), the process was interrupted via the running terminal session. Its traceback located the interruption in PyTorch backward execution. The shell sweep exited with code 130, so subsequent arms were not launched. No training result from this attempt is included in results.

A bounded, instrumented float32 CPU benchmark is being used to separate a precision/runtime issue from a fundamental resource limitation. This is engineering calibration; it does not alter test labels or select research outcomes.

Update: the three-step float32 calibration succeeded and the subsequent [float32 study](../local-pilot-fp32/RESULTS.md) completed all six fits and seven evaluations. The interrupted bfloat16 attempt remains excluded.
