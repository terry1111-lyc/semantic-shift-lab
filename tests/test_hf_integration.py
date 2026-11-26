"""Opt-in real library integration with a tiny RANDOM model; no research claims."""
import json
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("RUN_HF_INTEGRATION") == "1", "Set RUN_HF_INTEGRATION=1 after installing .[hf]")
class HFIntegrationTests(unittest.TestCase):
    def test_inference_lora_reload_and_provenance(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from semantic_shift.data import generate
        from semantic_shift.io import write_jsonl
        from semantic_shift.identity import directory_digest
        from semantic_shift.cli import run, score
        from semantic_shift.training import train
        torch.set_num_threads(1)
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "random-model"
            model_dir.mkdir()
            vocabulary = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[BOS]", "[EOS]", "user", "assistant", "FINAL", ":", "0", "1", "2", "3", "4"])}
            raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.chat_template = "{% for message in messages %}{{ '[BOS] ' + message['role'] + ' : ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ '[BOS] assistant : ' }}{% endif %}"
            tokenizer.save_pretrained(model_dir)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=512, n_embd=16, n_layer=1, n_head=2, bos_token_id=2, eos_token_id=3, pad_token_id=0))
            model.save_pretrained(model_dir)
            revision = directory_digest(model_dir)
            data = root / "data.jsonl"
            write_jsonl(data, generate(1))
            baseline = run(data, root / "baseline", backend="hf", model=str(model_dir), revision=revision, max_new_tokens=4)
            self.assertFalse(score(data, baseline, 20)["is_demo"])
            cfg = dict(model=str(model_dir), revision=revision, condition="E", device="cpu", seed=17, token_budget=150, max_steps=4, max_length=512, lora_rank=2, lora_alpha=4)
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            result = train(config, data, root / "adapter")
            self.assertGreater(result["steps"], 0)
            manifest = json.loads((root / "adapter" / "training_manifest.json").read_text())
            self.assertTrue(manifest["budget_reached"])
            adapted = run(data, root / "adapted", backend="hf", model=str(model_dir), revision=revision, adapter=str(root / "adapter"), max_new_tokens=4)
            score(data, adapted, 20)
            metadata = json.loads((adapted / "metadata.json").read_text())
            self.assertIsNotNone(metadata["adapter_sha256"])
            with self.assertRaises(ValueError):
                run(data, root / "wrong", backend="hf", model=str(model_dir), revision="0"*64, adapter=str(root / "adapter"))


if __name__ == "__main__":
    unittest.main()
