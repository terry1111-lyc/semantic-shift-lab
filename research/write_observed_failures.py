"""Deterministically extract examples and full-denominator diagnostics from actual outputs."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_shift.io import read_jsonl, save_json
from semantic_shift.evaluation import parse_answer

study = ROOT / 'research/local-pilot-fp32'
groups = defaultdict(dict)
for row in read_jsonl(study / 'data.jsonl'):
    if row['split'] == 'test':
        groups[row['group_id']][row['variant']] = row
names = ['A'] + [f'{arm}-seed{seed}' for seed in (17, 29) for arm in 'BCE']
diagnostics = []
examples = []
lines = ['# 真实失败案例 / Observed failures', '',
         '以下输出来自实际模型。案例是结果产生后的诊断选择，不是预先注册的独立检验；选择规则和完整分母同时公开。模板仅经助手检查，未做独立人工审核。', '',
         '## 改写失败与编辑后沿用答案', '',
         '“原题对→改写错”以原题正确的题组为分母；“编辑沿用”指原题正确、事实编辑后输出仍等于原题答案。后者只是行为描述，不能直接解释模型内部机制。', '',
         '| 运行 | 原题正确 / 60 | 其中改写答错 | 其中编辑沿用旧答案 |',
         '|---|---:|---:|---:|']
for name in names:
    pred = {p['id']: p for p in read_jsonl(study / name / 'predictions.jsonl')}
    correct_original = []
    rewrite_fail = []
    unchanged_edit = []
    for gid, group in groups.items():
        answer = {v: parse_answer(pred[r['id']]['raw_output']) for v, r in group.items()}
        if answer['original'] == group['original']['answer']:
            correct_original.append(gid)
            if answer['paraphrase'] != group['paraphrase']['answer']:
                rewrite_fail.append(gid)
            if answer['edit'] == answer['original'] != group['edit']['answer']:
                unchanged_edit.append(gid)
    diagnostics.append(dict(run=name, original_correct=len(correct_original),
                            paraphrase_fail_given_original_correct=len(rewrite_fail),
                            unchanged_edit_given_original_correct=len(unchanged_edit),
                            paraphrase_failure_groups=rewrite_fail,
                            unchanged_edit_groups=unchanged_edit))
    lines.append(f'| {name} | {len(correct_original)} / 60 | {len(rewrite_fail)} / {len(correct_original)} | {len(unchanged_edit)} / {len(correct_original)} |')
    # Stable selection rule: first matching group in frozen dataset order.
    candidates = rewrite_fail if name == 'B-seed17' else unchanged_edit if name == 'E-seed17' else []
    if candidates:
        gid = candidates[0]
        examples.append(dict(run=name, group_id=gid,
                             selection='first original-correct / paraphrase-wrong' if name == 'B-seed17' else 'first original-correct / unchanged-edit',
                             variants=[dict(**r, raw_output=pred[r['id']]['raw_output']) for r in groups[gid].values()]))
lines += ['', '零分母表示该条件无法估计比例，不是 0% 的条件错误率。', '']
for case in examples:
    lines += [f"## 案例：{case['run']} / `{case['group_id']}`", '',
              f"选择规则：`{case['selection']}`，按冻结数据顺序。", '']
    for row in case['variants']:
        lines += [f"**{row['variant']}**（正确答案：{row['answer']}）", '',
                  row['prompt'], '', '实际原始输出：', '', '```text', row['raw_output'], '```', '']
    lines += [f"证据：[完整预测]({case['run']}/predictions.jsonl)、[冻结数据](data.jsonl)。", '']
lines += ['## 按题型拆解', '',
          '每种题型只有一个测试模板、20 个题组。表内为“原题正确 / 四题全对”的百分比；不能把单个模板上的失败推广到整个题型。', '',
          '| 运行 | 卡片转移 transfer | 比较 comparison | 分组 groups |', '|---|---:|---:|---:|']
for name in names:
    summary = json.loads((study / name / 'summary.json').read_text())
    cells = []
    for family in ['transfer', 'comparison', 'groups']:
        entry = summary['by_family'][family]
        cells.append(f"{100*entry['original_accuracy']:.1f}% / {100*entry['group_all_correct']:.1f}%")
    lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
lines += ['', '## 测量与运行失败', '',
          '- [严格格式导致数学正确回答判错](../local-pilot/development/MEASUREMENT_FAILURE.md)：保留最初 12 条验证输出，未将事后宽松解析替代主指标。',
          '- [bfloat16 训练中断](../local-pilot/TRAINING_INTERRUPTION.md)：保留失败记录；随后完成的是显式 float32 新轮次。',
          '- A 在当前短答案协议下有大量格式失败和截断；A 到 SFT 的分数变化同时包含格式遵循和任务学习，不能全部解释为推理提升。',
          '- E 的事实编辑配对正确率在两个种子间从 1.7% 变为 20.0%，提示小预算结果不稳定。主指标 E−C 两种子分别为 0 与 +1.7 个百分点，尚无可靠优势。', '',
          '以上现象支持下一轮做更好的测量和控制实验；它们本身不证明某种内部机制、普遍权衡或新算法有效。', '']
save_json(study / 'observed-failures.json', {'diagnostics': diagnostics, 'examples': examples})
(study / 'OBSERVED_FAILURES.md').write_text('\n'.join(lines))
print(json.dumps({'diagnostics': [{k:v for k,v in d.items() if not k.endswith('_groups')} for d in diagnostics], 'examples':len(examples)}))
