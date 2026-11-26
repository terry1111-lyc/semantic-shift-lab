"""Analyze actual frozen pilot outputs; never substitutes demo results."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_shift.io import read_jsonl, save_json, fingerprint
from semantic_shift.evaluation import parse_answer


def paired_cluster_ci(values,templates,repeats=5000):
    groups=defaultdict(list)
    for v,t in zip(values,templates):groups[t].append(v)
    keys=sorted(groups)
    rng=random.Random(20261125)
    scores=[]
    for _ in range(repeats):
        draw=[v for k in rng.choices(keys,k=len(keys)) for v in groups[k]]
        scores.append(sum(draw)/len(draw))
    scores.sort()
    return [scores[int(.025*(repeats-1))],scores[int(.975*(repeats-1))]]


def analyze(root):
    root=Path(root)
    rows=[r for r in read_jsonl(root/'data.jsonl') if r['split']=='test']
    by_id={r['id']:r for r in rows}
    grouped=defaultdict(list)
    for r in rows:grouped[r['group_id']].append(r)
    names=['A']+[f'{arm}-seed{seed}' for seed in (17,29) for arm in 'BCE']
    table=[];outputs={};group_scores={};candidates=[]
    for name in names:
        run=root/name
        if not (run/'summary.json').exists(): continue
        meta=json.loads((run/'metadata.json').read_text())
        if meta['is_demo'] or meta['status']!='complete':raise ValueError('Only complete actual-model runs allowed')
        if meta['dataset_sha256']!=fingerprint(rows):raise ValueError('Data mismatch')
        pred=read_jsonl(run/'predictions.jsonl')
        if len(pred)!=len(rows) or {r['id'] for r in pred}!=set(by_id):raise ValueError('Incomplete predictions')
        outputs[name]={r['id']:r for r in pred}
        parsed={p['id']:parse_answer(p['raw_output']) for p in pred}
        correct={i:parsed[i]==r['answer'] if parsed[i] is not None else False for i,r in by_id.items()}
        group_scores[name]={gid:int(all(correct[r['id']] for r in group)) for gid,group in grouped.items()}
        valid=sum(v is not None for v in parsed.values())
        summary=json.loads((run/'summary.json').read_text())
        entry=dict(run=name,**{k:v['value'] for k,v in summary['metrics'].items()},
                   all_item_accuracy=sum(correct.values())/len(rows),parseable=valid,total_items=len(rows),
                   format_failure_rate=1-valid/len(rows),
                   numeric_accuracy_among_parseable=sum(correct.values())/valid if valid else None,
                   token_cap_fraction=meta['token_cap_fraction'],inference_seconds=meta['inference_seconds'],
                   generated_tokens=meta['generated_tokens'])
        manifest=root/'training'/f'{name}-manifest.json'
        if manifest.exists():
            training=json.loads(manifest.read_text())
            entry.update(training_seconds=training['elapsed_seconds'],training_steps=training['steps'],
                         processed_tokens=training['processed_tokens'],supervised_tokens=training['supervised_tokens'],
                         unique_examples_seen=training['unique_examples_seen'])
        table.append(entry)
        for r in rows:
            if not correct[r['id']]:
                candidates.append(dict(run=name,id=r['id'],prompt=r['prompt'],gold=r['answer'],
                                       variant=r['variant'],raw_output=outputs[name][r['id']]['raw_output'],
                                       parsed=parsed[r['id']],reached_token_cap=outputs[name][r['id']]['reached_token_cap']))
    comparisons=[]
    gids=sorted(grouped)
    templates=[grouped[g][0]['template_id'] for g in gids]
    for seed in (17,29):
        left,right=f'E-seed{seed}',f'C-seed{seed}'
        if left in group_scores and right in group_scores:
            deltas=[group_scores[left][g]-group_scores[right][g] for g in gids]
            comparisons.append(dict(comparison=f'{left} minus {right}',endpoint='group_all_correct',
                                    delta=sum(deltas)/len(deltas),exploratory_template_cluster_95ci=paired_cluster_ci(deltas,templates)))
    result=dict(status='ACTUAL MODEL PILOT',runs_completed=len(table),expected_runs=7,
                test_groups=len(grouped),test_templates=len(set(templates)),table=table,paired_comparisons=comparisons,
                caveat='Three template clusters and two training seeds; exploratory only. Format and truncation confounds must be considered. No diversity-matched baseline.')
    save_json(root/'analysis.json',result)
    save_json(root/'audit-candidates.json',candidates)
    lines=['# Actual-model pilot: quantitative results','', '**These are actual Qwen3-0.6B outputs, not the repository demo fixtures.**','',
           '| Run | Original | Invariant all-correct | Edit-pair correct | All-four correct | Format failures | Token cap |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| '+r['run']+' | '+' | '.join(f"{100*r[k]:.1f}%" for k in ['original_accuracy','invariant_all_correct','edit_pair_correct','group_all_correct','format_failure_rate','token_cap_fraction'])+' |')
    lines+=['','## Primary paired comparison','', 'E minus C on the same test groups, separately for each seed. Intervals resample only three template clusters and are exploratory.']
    for r in comparisons:
        lo,hi=r['exploratory_template_cluster_95ci']
        lines.append(f"- {r['comparison']}: {100*r['delta']:+.1f} percentage points; interval [{100*lo:+.1f}, {100*hi:+.1f}].")
    lines+=['','## Interpretation limits','',result['caveat'],
            'Answer-only generation is capped at 16 new tokens. A format failure or truncated answer cannot by itself establish a reasoning failure. The original longer development run is preserved separately.',
            'Conditional accuracy among parseable outputs is a diagnostic with unequal denominators, not a fair replacement for the primary metric.',
            'All runs share the same model snapshot, test set and decoding. Training uses a small matched total-token budget but different unique-example distributions. The study does not test RL or longer reasoning.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        colors={'A':'#677581','B':'#4477AA','C':'#EE7733','E':'#228833'}
        for i,r in enumerate(table):
            color=colors[r['run'][0]]
            axes[0].bar(i-.18,100*r['original_accuracy'],width=.36,color=color,alpha=.35,label='Original correct' if i==0 else None)
            axes[0].bar(i+.18,100*r['group_all_correct'],width=.36,color=color,label='All four correct' if i==0 else None)
            axes[1].bar(i-.18,100*r['format_failure_rate'],width=.36,color=color,label='Format failure' if i==0 else None)
            axes[1].bar(i+.18,100*r['token_cap_fraction'],width=.36,color=color,alpha=.4,label='Token cap' if i==0 else None)
        for ax in axes:
            ax.set_xticks(range(len(table)),[r['run'] for r in table],rotation=35,ha='right')
            ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_title('Original accuracy versus grouped reliability');axes[0].legend(frameon=False)
        axes[1].set_title('Output-format and truncation diagnostics');axes[1].legend(frameon=False)
        fig.suptitle('Actual Qwen3-0.6B local pilot · 60 groups / 3 templates · exploratory',fontsize=12)
        fig.tight_layout()
        fig.savefig(root/'results.png',dpi=180)
        fig.savefig(root/'results.svg')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for entry in table:
            history=root/'training'/f"{entry['run']}-losses.json"
            if not history.exists():continue
            points=json.loads(history.read_text())
            smooth=[sum(p['loss'] for p in points[max(0,i-9):i+1])/len(points[max(0,i-9):i+1]) for i in range(len(points))]
            ax.plot([p['processed_tokens'] for p in points],smooth,label=entry['run'],color=colors[entry['run'][0]],linestyle='--' if '29' in entry['run'] else '-')
        if ax.lines:
            ax.set_xlabel('Processed training tokens');ax.set_ylabel('Training loss (10-step moving mean)')
            ax.set_title('Optimization diagnostic — not generalization evidence')
            ax.legend(frameon=False,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(root/'training-loss.png',dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(Path(__file__).parent/'local-pilot'))
    analyze(p.parse_args().study)
