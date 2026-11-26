import copy
import json
import tempfile
import unittest
from pathlib import Path
from semantic_shift.data import generate, validate, solve, template
from semantic_shift.evaluation import parse_answer, evaluate, bootstrap
from semantic_shift.io import fingerprint, write_jsonl, read_jsonl
from semantic_shift.models import demo_predictions
from semantic_shift.training import select_training, supervised_tokens
from semantic_shift.cli import main, run, score


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = generate(3)

    def test_deterministic_and_seeded(self):
        self.assertEqual(self.rows, generate(3))
        self.assertNotEqual(self.rows, generate(3, 18))

    def test_all_templates_have_correct_programs(self):
        a, b, c = 50, 7, 3
        expected = {("transfer",0):(54,40), ("transfer",1):(40,46), ("transfer",2):(60,54),
                    ("comparison",0):(57,43), ("comparison",1):(60,54), ("comparison",2):(107,93),
                    ("groups",0):(71,74), ("groups",1):(329,371), ("groups",2):(347,329)}
        for (family,index),(original,edited) in expected.items():
            texts, p, e = template(family,index,a,b,c)
            self.assertEqual(len(texts),4)
            self.assertEqual(solve(p),original)
            self.assertEqual(solve(e),edited)

    def test_disjoint_templates(self):
        sets = [{r["template_id"] for r in self.rows if r["split"] == split} for split in ("train","dev","test")]
        self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
        self.assertEqual(validate(self.rows)["groups"],27)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): validate(self.rows+[self.rows[0]])

    def test_missing_variant_rejected(self):
        with self.assertRaises(ValueError): validate(self.rows[:-1])

    def test_mixed_group_split_rejected(self):
        self.rows[0]["split"]="test"
        with self.assertRaises(ValueError): validate(self.rows)

    def test_template_leakage_rejected(self):
        for row in self.rows[:4]: row["split"]="test"
        with self.assertRaises(ValueError): validate(self.rows)

    def test_wrong_gold_rejected(self):
        self.rows[0]["answer"] += 1
        with self.assertRaises(ValueError): validate(self.rows)

    def test_invariant_program_change_rejected(self):
        self.rows[1]["program"] = 999
        self.rows[1]["answer"] = 999
        with self.assertRaises(ValueError): validate(self.rows)

    def test_unchanged_edit_rejected(self):
        self.rows[3]["program"] = self.rows[0]["program"]
        self.rows[3]["answer"] = self.rows[0]["answer"]
        with self.assertRaises(ValueError): validate(self.rows)

    def test_invalid_sizes_and_program(self):
        for n in (0,1001,-1):
            with self.assertRaises(ValueError): generate(n)
        with self.assertRaises(ValueError): solve(["eval",1,2])


class ParseTests(unittest.TestCase):
    def test_supported_formats(self):
        for text, expected in [("20",20),("FINAL: -3",-3),("calculation 10+10=20\nFINAL: 20",20),
                               ("FINAL: 1,200.0",1200),("FINAL: 2\nFINAL: 3",3)]:
            self.assertEqual(parse_answer(text),expected)

    def test_ambiguous_outputs_not_guessed(self):
        for text in ("20 or 21", "twenty", "The answer is 20", "FINAL: 2.5", "FINAL: 1,20", "", "FINAL: NaN", "FINAL: 20 cards"):
            self.assertIsNone(parse_answer(text))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows=[r for r in generate(2) if r["split"]=="test"]
        self.pred=[dict(id=r["id"],raw_output=f"FINAL: {r['answer']}") for r in self.rows]
        self.meta=dict(dataset_sha256=fingerprint(self.rows),is_demo=True,model="test",backend="demo")

    def test_perfect(self):
        summary, errors=evaluate(self.rows,self.pred,self.meta,100)
        self.assertTrue(all(m["value"]==1 for m in summary["metrics"].values()))
        self.assertEqual(errors,[])
        self.assertTrue(summary["is_demo"])

    def test_invariant_wrong_is_not_success(self):
        for p in self.pred: p["raw_output"]="FINAL: -999"
        summary, _=evaluate(self.rows,self.pred,self.meta,100)
        self.assertEqual(summary["metrics"]["invariant_all_correct"]["value"],0)

    def test_changed_but_wrong_is_not_success(self):
        for i,row in enumerate(self.rows):
            if row["variant"]=="edit": self.pred[i]["raw_output"]="FINAL: -999"
        summary,_=evaluate(self.rows,self.pred,self.meta,100)
        self.assertEqual(summary["metrics"]["original_accuracy"]["value"],1)
        self.assertEqual(summary["metrics"]["edit_pair_correct"]["value"],0)

    def test_format_failure_counted(self):
        self.pred[0]["raw_output"]="twenty"
        summary,errors=evaluate(self.rows,self.pred,self.meta,100)
        self.assertEqual(summary["format_failure_rate"],1/len(self.rows))
        self.assertEqual(errors[0]["category"],"format_or_parser")

    def test_exact_prediction_coverage(self):
        for bad in (self.pred[:-1],self.pred+[self.pred[0]],self.pred+[dict(id="extra",raw_output="2")]):
            with self.assertRaises(ValueError): evaluate(self.rows,bad,self.meta,10)

    def test_fingerprint_required(self):
        self.meta["dataset_sha256"]="wrong"
        with self.assertRaises(ValueError): evaluate(self.rows,self.pred,self.meta,10)

    def test_explicit_demo_status_required(self):
        del self.meta["is_demo"]
        with self.assertRaises(ValueError): evaluate(self.rows,self.pred,self.meta,10)

    def test_cluster_bootstrap(self):
        self.assertIsNone(bootstrap([1,0],["a","a"],100))
        ci=bootstrap([1,1,0,0],["a","a","b","b"],100)
        self.assertEqual(ci,[0,1])
        self.assertEqual(ci,bootstrap([1,1,0,0],["a","a","b","b"],100))


class TrainingTests(unittest.TestCase):
    def test_selection_never_uses_dev_or_test(self):
        rows=generate(2)
        expected={"B":6,"C":18,"D":12,"E":24}
        for condition,n in expected.items():
            chosen=select_training(rows,condition)
            self.assertEqual(len(chosen),n)
            self.assertTrue(all(r["split"]=="train" for r in chosen))

    def test_unknown_condition_rejected(self):
        with self.assertRaises(ValueError): select_training(generate(1),"Z")

    def test_prompt_masking_and_truncation_guard(self):
        class FakeTokenizer:
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
                return "USER:"+msgs[0]["content"]+"ASSISTANT:"+(msgs[1]["content"] if len(msgs)>1 else "")
            def __call__(self,text,**kwargs): return {"input_ids":[ord(c) for c in text]}
        row=generate(1)[0]
        ids,labels=supervised_tokens(FakeTokenizer(),row,2000)
        start=next(i for i,label in enumerate(labels) if label!=-100)
        self.assertGreater(start,0)
        self.assertEqual(ids[start:],labels[start:])
        _, short_labels=supervised_tokens(FakeTokenizer(),row,2000,prompt_instruction="Integer only.",answer_format="integer_only")
        self.assertEqual("".join(chr(t) for t in short_labels if t!=-100),str(row["answer"]))
        with self.assertRaises(ValueError): supervised_tokens(FakeTokenizer(),row,5)


class PipelineTests(unittest.TestCase):
    def test_demo_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"demo"
            main(["demo","--output",str(dest)])
            report=(dest/"fault-injector"/"report.md").read_text()
            self.assertIn("DEMONSTRATION ONLY",report)
            with self.assertRaises(SystemExit): main(["demo","--output",str(dest)])

    def test_source_tampering_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/"data.jsonl"
            rows=generate(2); write_jsonl(path,rows)
            out=run(path,root/"run")
            rows[0]["prompt"] += " "
            write_jsonl(path,rows)
            with self.assertRaises(ValueError): score(path,out)
            meta=json.loads((out/"metadata.json").read_text()); meta["status"]="running"
            (out/"metadata.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError): score(path,out)

    def test_export_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/"data.jsonl"
            write_jsonl(path,generate(2))
            main(["export-sft","--data",str(path),"--condition","E","--output",str(root/"sft.jsonl")])
            self.assertEqual(len(read_jsonl(root/"sft.jsonl")),24)

    def test_demo_faults_are_explicit(self):
        rows=generate(20)
        predictions=demo_predictions(rows)
        faults={p["injected_fault"] for p in predictions}
        self.assertTrue({"ignored_fact_edit","rewrite_failure","format_failure"} <= faults)
        self.assertEqual(predictions,demo_predictions(rows))


if __name__ == "__main__":
    unittest.main()
