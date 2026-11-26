import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from .data import validate
from .io import fingerprint

# Deliberately strict: ambiguous prose is sent to manual review, not guessed.
NUMBER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"


def parse_answer(text):
    if not isinstance(text, str):
        return None
    candidates = re.findall(r"(?im)^\s*FINAL\s*:\s*(.*?)\s*$", text)
    if candidates:
        value = candidates[-1]
    elif "FINAL" in text.upper():
        return None
    else:
        value = text.strip()
    if not re.fullmatch(NUMBER, value):
        return None
    try:
        result = Decimal(value.replace(",", ""))
        return int(result) if result == result.to_integral_value() else None
    except InvalidOperation:
        return None


def bootstrap(values, clusters, repeats=1000, seed=17):
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    keys = sorted(grouped)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        drawn = [v for key in rng.choices(keys, k=len(keys)) for v in grouped[key]]
        samples.append(sum(drawn) / len(drawn))
    samples.sort()
    return [samples[int(.025 * (repeats-1))], samples[int(.975 * (repeats-1))]]


def evaluate(rows, predictions, meta, repeats=1000):
    validation = validate(rows)
    if meta.get("dataset_sha256") != fingerprint(rows):
        raise ValueError("Prediction metadata does not match this dataset")
    if not isinstance(meta.get("is_demo"), bool):
        raise ValueError("Metadata must explicitly declare is_demo")
    ids = [p.get("id") for p in predictions]
    expected = {r["id"] for r in rows}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Predictions must match every dataset ID exactly once")
    lookup = {p["id"]: p for p in predictions}
    groups = defaultdict(dict)
    errors = []
    formats = 0
    tokens = []
    for r in rows:
        p = lookup[r["id"]]
        if not isinstance(p.get("raw_output"), str):
            raise ValueError("Every prediction needs raw_output text")
        value = parse_answer(p["raw_output"])
        ok = value == r["answer"] if value is not None else False
        formats += value is None
        groups[r["group_id"]][r["variant"]] = (ok, r)
        token_count = p.get("output_tokens")
        if token_count is not None:
            if type(token_count) is not int or token_count < 0:
                raise ValueError("Invalid token count")
            tokens.append(token_count)
        if not ok:
            errors.append(dict(id=r["id"], group_id=r["group_id"], template_id=r["template_id"],
                               variant=r["variant"], prompt=r["prompt"], expected=r["answer"],
                               parsed=value, raw_output=p["raw_output"],
                               category="format_or_parser" if value is None else "wrong_answer",
                               human_diagnosis="pending", is_demo=meta["is_demo"]))
    vectors = defaultdict(list)
    clusters = []
    family_results = defaultdict(list)
    for gid in sorted(groups):
        group = groups[gid]
        o, p, r, e = [group[v][0] for v in ("original", "paraphrase", "reorder", "edit")]
        values = {"original_accuracy": o, "invariant_all_correct": o and p and r,
                  "edit_pair_correct": o and e, "group_all_correct": o and p and r and e}
        clusters.append(group["original"][1]["template_id"])
        family_results[group["original"][1]["family"]].append(values)
        for key, value in values.items():
            vectors[key].append(int(value))
    metrics = {key: {"value": sum(v)/len(v),
                     "template_cluster_bootstrap_95ci": bootstrap(v, clusters, repeats)}
               for key, v in vectors.items()}
    return dict(is_demo=meta["is_demo"], status="DEMONSTRATION ONLY" if meta["is_demo"] else "MODEL RUN; HUMAN AUDIT REQUIRED",
                dataset_sha256=fingerprint(rows), validation=validation, metrics=metrics,
                format_failure_rate=formats/len(rows), output_tokens_total=sum(tokens) if tokens else None,
                token_coverage=len(tokens)/len(rows),
                by_family={f: {k: sum(x[k] for x in vs)/len(vs) for k in metrics} for f, vs in family_results.items()},
                uncertainty_note="Exploratory percentile bootstrap over template IDs; few templates give unreliable population uncertainty. Not a training-seed CI.",
                model=meta.get("model"), backend=meta.get("backend")), errors


def markdown_report(summary):
    lines = ["# Semantic Shift Lab — run report", "", "**"+summary["status"]+"**", "",
             "No causal or publication claim is implied by this report.", "",
             f"Model: `{summary['model']}`; backend: `{summary['backend']}`", "",
             "| Metric | Value | Exploratory template-bootstrap 95% interval |",
             "|---|---:|---:|"]
    for name, metric in summary["metrics"].items():
        ci = metric["template_cluster_bootstrap_95ci"]
        interval = "unavailable" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {name} | {metric['value']:.3f} | {interval} |")
    lines += ["", f"Format/parser failure rate: {summary['format_failure_rate']:.3f}",
              f"Unreviewed records: {summary['validation']['unreviewed_records']}", "", summary["uncertainty_note"],
              "", "See errors.jsonl for raw responses. Error categories are not mechanistic explanations.", ""]
    return "\n".join(lines)
