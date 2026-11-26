import hashlib

INSTRUCTION = "Solve the problem. Give a brief calculation, then put the integer answer alone on the last line as FINAL: <integer>."


def messages(prompt, instruction=INSTRUCTION):
    return [{"role": "user", "content": instruction + "\n\n" + prompt}]


def demo_predictions(rows, seed=17):
    """Label-aware fault injection ONLY. Never use this to estimate model quality."""
    originals = {r["group_id"]: r["answer"] for r in rows if r["variant"] == "original"}
    result = []
    for r in rows:
        bucket = int(hashlib.sha256(f"{seed}/{r['group_id']}".encode()).hexdigest()[:8], 16) % 5
        answer, fault = r["answer"], None
        if bucket == 0 and r["variant"] == "edit":
            answer, fault = originals[r["group_id"]], "ignored_fact_edit"
        elif bucket == 1 and r["variant"] == "paraphrase":
            answer, fault = answer + 1, "rewrite_failure"
        output = f"FINAL: {answer}"
        if bucket == 2 and r["variant"] == "reorder":
            output, fault = f"The answer is {answer}.", "format_failure"
        result.append(dict(id=r["id"], raw_output=output, output_tokens=None,
                           latency_seconds=None, injected_fault=fault))
    return result


class HuggingFaceModel:
    def __init__(self, model, revision, adapter=None, device="cpu"):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("Choose a tokenizer with a chat template")
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, trust_remote_code=False, dtype=torch.float32)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(device).eval()
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, prompt, max_new_tokens=256):
        # Receive only prompt text: no gold label, program, or variant metadata.
        import time
        text = self.tokenizer.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        def sync():
            if self.model.device.type == "cuda":
                self.torch.cuda.synchronize()
            elif self.model.device.type == "mps":
                self.torch.mps.synchronize()
        sync()
        start = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad)
        sync()
        generated = out[0, inputs["input_ids"].shape[1]:]
        return dict(raw_output=self.tokenizer.decode(generated, skip_special_tokens=True),
                    output_tokens=len(generated), input_tokens=inputs["input_ids"].shape[1],
                    latency_seconds=time.perf_counter()-start,
                    reached_token_cap=len(generated) == max_new_tokens)
