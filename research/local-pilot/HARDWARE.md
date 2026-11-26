# Local execution

Apple M1, 16 GB unified memory. PyTorch reports MPS built but unavailable in this execution context. CPU uses four threads; main evaluation batches eight independent conversations with left padding. No server was rented.

The development answer-only check was repeated with batch size 1. Most outputs matched batch size 4; one long output differed before truncation. Batch shape can affect floating-point greedy decoding. All compared arms use the same batch ordering and size.
