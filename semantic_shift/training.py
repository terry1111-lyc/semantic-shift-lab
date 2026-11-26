"""Optional, single-device LoRA SFT. Requires the hf extra and a compatible chat model."""
import json
import random
from pathlib import Path
from .data import validate, expression
from .io import read_jsonl, fingerprint, save_json
from .models import messages
from .identity import validate_model_identity

CONDITIONS = {
    "B": ("original",),
    "C": ("original", "paraphrase", "reorder"),
    "D": ("original", "edit"),
    "E": ("original", "paraphrase", "reorder", "edit"),
}


def select_training(rows, condition):
    validate(rows)
    if condition not in CONDITIONS:
        raise ValueError("Condition must be B, C, D, or E")
    chosen = [r for r in rows if r["split"] == "train" and r["variant"] in CONDITIONS[condition]]
    if not chosen:
        raise ValueError("No training rows")
    return chosen


def supervised_tokens(tokenizer, row, max_length, chat_template_kwargs=None, prompt_instruction=None, answer_format="calculation"):
    chat_template_kwargs = chat_template_kwargs or {}
    conversation = messages(row["prompt"]) if prompt_instruction is None else messages(row["prompt"], prompt_instruction)
    answer = f"{expression(row['program'])} = {row['answer']}\nFINAL: {row['answer']}"
    if answer_format == "integer_only":
        answer = str(row["answer"])
    elif answer_format != "calculation":
        raise ValueError("Unknown answer format")
    prefix = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
    full = tokenizer.apply_chat_template(conversation + [{"role": "assistant", "content": answer}],
                                          tokenize=False, add_generation_prompt=False, **chat_template_kwargs)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat template prefix is not token-aligned; cannot safely mask prompt tokens")
    if len(ids) > max_length:
        raise ValueError("Example exceeds max_length; increase it instead of silently truncating labels")
    if len(ids) <= len(prefix_ids):
        raise ValueError("No supervised answer tokens")
    return ids, [-100] * len(prefix_ids) + ids[len(prefix_ids):]


def train(config_path, data_path, output):
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model
    cfg = json.loads(Path(config_path).read_text())
    allowed = {"model", "revision", "condition", "seed", "device", "max_length", "learning_rate",
               "token_budget", "max_steps", "lora_rank", "lora_alpha", "target_modules", "chat_template_kwargs", "prompt_instruction", "answer_format", "dtype"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown config keys: {set(cfg)-allowed}")
    if not cfg.get("model") or cfg["model"].startswith("REPLACE"):
        raise ValueError("Set model and an immutable model revision in the config")
    revision = cfg.get("revision", "")
    validate_model_identity(cfg["model"], revision)
    seed = cfg.get("seed", 17)
    budget, max_steps = cfg.get("token_budget", 100000), cfg.get("max_steps", 10000)
    if budget < 1 or max_steps < 1:
        raise ValueError("Positive budgets required")
    output = Path(output)
    if output.exists():
        raise ValueError("Training output must be a new directory")
    rows = read_jsonl(data_path)
    selected = select_training(rows, cfg["condition"])
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=revision, trust_remote_code=False)
    encoded = [supervised_tokens(tokenizer, r, cfg.get("max_length", 512), cfg.get("chat_template_kwargs"), cfg.get("prompt_instruction"), cfg.get("answer_format", "calculation")) for r in selected]
    dtype_name = cfg.get("dtype", "float32")
    if dtype_name not in {"float32", "bfloat16", "float16"}:
        raise ValueError("Unsupported dtype")
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=revision, trust_remote_code=False, dtype=getattr(torch, dtype_name))
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=cfg.get("lora_rank", 8),
                              lora_alpha=cfg.get("lora_alpha", 16), lora_dropout=0.0,
                              target_modules=cfg.get("target_modules", "all-linear")))
    model.to(cfg.get("device", "cpu")).train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.get("learning_rate", 0.0001), weight_decay=0.0)
    rng = random.Random(seed)
    order = list(range(len(encoded)))
    seen, total_tokens, supervised, steps, log = set(), 0, 0, 0, []
    while total_tokens < budget and steps < max_steps:
        rng.shuffle(order)
        for index in order:
            ids, labels = encoded[index]
            x = torch.tensor([ids], device=model.device)
            y = torch.tensor([labels], device=model.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_tokens += len(ids)
            supervised += sum(t != -100 for t in labels)
            steps += 1
            seen.add(selected[index]["id"])
            log.append(dict(step=steps, loss=float(loss.detach().cpu()), processed_tokens=total_tokens))
            if steps % 10 == 0:
                print(json.dumps(dict(phase="training", step=steps, processed_tokens=total_tokens, loss=log[-1]["loss"])), flush=True)
            if total_tokens >= budget or steps >= max_steps:
                break
    output.mkdir(parents=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    save_json(output / "training_manifest.json", dict(config=cfg, dataset_sha256=fingerprint(rows),
              selected_sha256=fingerprint(selected), examples_available=len(selected), unique_examples_seen=len(seen),
              steps=steps, processed_tokens=total_tokens, supervised_tokens=supervised,
              budget_reached=total_tokens >= budget, token_budget_overshoot=max(0, total_tokens-budget),
              versions={"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              warning="Pilot SFT, not a controlled publication result. See protocol for diversity controls and multiple seeds."))
    save_json(output / "losses.json", log)
    return dict(output=str(output), steps=steps, processed_tokens=total_tokens)
