"""Original, deterministic English templates. Pilot data, not a validated benchmark."""
import random
from collections import defaultdict

VARIANTS = ("original", "paraphrase", "reorder", "edit")
SPLITS = {0: "train", 1: "dev", 2: "test"}


def solve(program):
    if type(program) is int:
        return program
    if not isinstance(program, list) or len(program) != 3 or program[0] not in {"+", "-", "*"}:
        raise ValueError("Invalid arithmetic program")
    op, left, right = program
    a, b = solve(left), solve(right)
    return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()


def expression(p):
    return str(p) if type(p) is int else f"({expression(p[1])} {p[0]} {expression(p[2])})"


def template(family, index, a, b, c):
    if family == "transfer" and index == 0:
        texts = [f"Lin has {a} cards, receives {b} cards, then gives away {c} cards. How many cards remain?",
                 f"Starting with {a} cards, Lin is given {b} more before handing {c} to a friend. How many does Lin keep?",
                 f"Lin gives away {c} cards after receiving {b}. Before either event, Lin had {a} cards. How many remain?",
                 f"Lin has {a} cards, gives away {b} cards, then gives away {c} cards. How many cards remain?"]
        return texts, ["-", ["+", a, b], c], ["-", ["-", a, b], c]
    if family == "transfer" and index == 1:
        texts = [f"Lin has {a} cards and gives away {b}. Lin then gives away {c}. How many cards remain?",
                 f"From {a} cards, Lin hands {b} to one friend and {c} to another. How many are left?",
                 f"The second gift is {c} cards and the first is {b}. Lin started with {a} cards and gives both gifts. How many remain?",
                 f"Lin has {a} cards and gives away {b}. Lin then receives {c}. How many cards remain?"]
        return texts, ["-", ["-", a, b], c], ["+", ["-", a, b], c]
    if family == "transfer" and index == 2:
        texts = [f"Lin has {a} cards, receives {b} cards, then receives {c} cards. How many cards does Lin have?",
                 f"Lin adds two gifts of {b} and {c} cards to an initial collection of {a}. How many cards are there now?",
                 f"A second gift contains {c} cards and the first contains {b}. Lin had {a} cards before receiving both. How many now?",
                 f"Lin has {a} cards, receives {b} cards, then gives away {c} cards. How many cards does Lin have?"]
        return texts, ["+", ["+", a, b], c], ["-", ["+", a, b], c]
    if family == "comparison" and index == 0:
        texts = [f"Mia has {a} books. Noah has {b} more books than Mia. How many books does Noah have?",
                 f"Mia owns {a} books, which is {b} fewer than Noah owns. How many does Noah own?",
                 f"Noah owns {b} more books than Mia. Mia owns {a}. How many does Noah own?",
                 f"Mia has {a} books. Noah has {b} fewer books than Mia. How many books does Noah have?"]
        return texts, ["+", a, b], ["-", a, b]
    if family == "comparison" and index == 1:
        texts = [f"Mia has {a} books. Noah has {b} more than Mia. Ava has {c} more than Noah. How many does Ava have?",
                 f"Ava owns {c} more books than Noah, who owns {b} more than Mia's {a} books. How many books does Ava own?",
                 f"Ava has {c} more books than Noah. Mia has {a}. Noah has {b} more than Mia. How many does Ava have?",
                 f"Mia has {a} books. Noah has {b} more than Mia. Ava has {c} fewer than Noah. How many does Ava have?"]
        return texts, ["+", ["+", a, b], c], ["-", ["+", a, b], c]
    if family == "comparison" and index == 2:
        texts = [f"Mia has {a} books. Noah has {b} more than Mia. How many books do they have together?",
                 f"With {a} books belonging to Mia and {b} additional books in Noah's collection compared with hers, what is their combined total?",
                 f"Noah has {b} more books than Mia. Together, how many books do they have if Mia has {a}?",
                 f"Mia has {a} books. Noah has {b} fewer than Mia. How many books do they have together?"]
        return texts, ["+", a, ["+", a, b]], ["+", a, ["-", a, b]]
    if family == "groups" and index == 0:
        texts = [f"There are {b} boxes with {c} balls in each box, and {a} loose balls. How many balls are there in total?",
                 f"Count all the balls: {a} are loose, and each of {b} boxes contains {c}. What is the total?",
                 f"There are {a} loose balls. Each box contains {c} balls, and there are {b} boxes. How many balls in total?",
                 f"There are {b+1} boxes with {c} balls in each box, and {a} loose balls. How many balls are there in total?"]
        return texts, ["+", ["*", b, c], a], ["+", ["*", b+1, c], a]
    if family == "groups" and index == 1:
        texts = [f"Each of {b} boxes contains {a} balls. Exactly {c} balls are removed from each box. How many balls remain across all boxes?",
                 f"There are {b} boxes, each starting with {a} balls and losing {c}. How many balls are left altogether?",
                 f"Exactly {c} balls are removed from every box. There are {b} boxes and each initially holds {a}. How many remain altogether?",
                 f"Each of {b} boxes contains {a} balls. Exactly {c} balls are added to each box. How many balls are now across all boxes?"]
        return texts, ["*", b, ["-", a, c]], ["*", b, ["+", a, c]]
    if family == "groups" and index == 2:
        texts = [f"There are {b} boxes, each containing {a} balls. Exactly {c} balls are removed in total. How many balls remain?",
                 f"From {b} boxes holding {a} balls apiece, a combined total of {c} balls is taken out. How many balls remain altogether?",
                 f"Exactly {c} balls are taken out overall. Initially there were {b} boxes with {a} balls in each. How many balls remain?",
                 f"There are {b} boxes, each containing {a} balls. Exactly {c} balls are removed from each box. How many balls remain?"]
        return texts, ["-", ["*", b, a], c], ["*", b, ["-", a, c]]
    raise ValueError("Unknown template")


def generate(per_template=20, seed=17):
    if not 1 <= per_template <= 1000:
        raise ValueError("per_template must be 1..1000")
    rng = random.Random(seed)
    rows = []
    for family in ("transfer", "comparison", "groups"):
        for index in range(3):
            tid = f"{family}_{index}"
            seen = set()
            for n in range(per_template):
                while True:
                    a, b, c = rng.randint(30, 99), rng.randint(2, 12), rng.randint(2, 12)
                    texts, original, edited = template(family, index, a, b, c)
                    if texts[0] not in seen:
                        seen.add(texts[0]); break
                gid = f"s{seed}-{tid}-{n:04d}"
                for variant, prompt in zip(VARIANTS, texts):
                    program = edited if variant == "edit" else original
                    rows.append(dict(id=f"{gid}/{variant}", group_id=gid, family=family,
                                     template_id=tid, split=SPLITS[index], variant=variant,
                                     prompt=prompt, program=program, answer=solve(program),
                                     review_status="unreviewed", schema_version=1))
    validate(rows)
    return rows


def validate(rows):
    if not rows:
        raise ValueError("Empty dataset")
    ids, prompts, groups, templates = set(), {}, defaultdict(list), defaultdict(set)
    for r in rows:
        required = {"id", "group_id", "family", "template_id", "split", "variant", "prompt", "program", "answer"}
        if not required <= r.keys():
            raise ValueError("Missing data fields")
        if r["id"] in ids:
            raise ValueError("Duplicate ID")
        ids.add(r["id"])
        if r["split"] not in {"train", "dev", "test"} or r["variant"] not in VARIANTS:
            raise ValueError("Invalid split or variant")
        if type(r["answer"]) is not int or solve(r["program"]) != r["answer"]:
            raise ValueError("Answer/program mismatch")
        if not isinstance(r["prompt"], str) or not r["prompt"].strip():
            raise ValueError("Empty prompt")
        normalized = " ".join(r["prompt"].lower().split())
        if normalized in prompts and prompts[normalized] != r["split"]:
            raise ValueError("Prompt leakage across splits")
        prompts[normalized] = r["split"]
        templates[r["template_id"]].add(r["split"])
        groups[r["group_id"]].append(r)
    if any(len(s) != 1 for s in templates.values()):
        raise ValueError("Template leakage across splits")
    for group in groups.values():
        if len(group) != 4 or {r["variant"] for r in group} != set(VARIANTS):
            raise ValueError("Each group needs exactly four variants")
        for field in ("split", "template_id", "family"):
            if len({r[field] for r in group}) != 1:
                raise ValueError("Inconsistent group metadata")
        answers = {r["variant"]: r["answer"] for r in group}
        if not answers["original"] == answers["paraphrase"] == answers["reorder"]:
            raise ValueError("Invariant answers differ")
        if answers["edit"] == answers["original"]:
            raise ValueError("Edit must change the answer")
    return {"records": len(rows), "groups": len(groups), "templates": len(templates),
            "unreviewed_records": sum(r.get("review_status") != "approved" for r in rows)}
