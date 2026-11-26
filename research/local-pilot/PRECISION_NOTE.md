# Precision verification

Before completing the first test run, an independent load of the fixed model snapshot under Transformers 5.24 showed that the default parameter dtype is torch.bfloat16. All study inference and frozen base weights use this checkpoint precision. PEFT may upcast trainable adapter parameters to float32. No quantization is used.

A development-only float32 check on four inputs produced the same output texts as the bfloat16 batch. Its 11.67-second runtime was measured while another process was running and cannot establish a speed comparison. This check is excluded from the primary study.

The original project README described full-precision loading too loosely; it is corrected to unquantized checkpoint precision. The pinned library versions and actual dtype must be retained when reproducing results.
