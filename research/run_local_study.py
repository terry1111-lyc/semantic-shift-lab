"""Reproduce the frozen local pilot. Model weights and adapters stay outside the release."""
import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_shift.data import validate
from semantic_shift.io import read_jsonl, save_json, write_jsonl, fingerprint
from semantic_shift.models import messages
from semantic_shift.identity import directory_digest
from semantic_shift.evaluation import evaluate, markdown_report


PROMPT = "Return only the integer answer to the problem. Do not explain. Do not add units or formatting."

def infer(model_path, revision, rows, output, adapter=None, batch_size=4, max_new_tokens=16):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    output=Path(output)
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    output.mkdir(parents=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=False,dtype=torch.float32).to('cpu').eval()
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter).eval()
    model.config.use_cache=True
    texts=[tokenizer.apply_chat_template(messages(r['prompt'],PROMPT),tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    meta=dict(is_demo=False,backend='hf-cpu-batched',model='Qwen/Qwen3-0.6B',revision=revision,
              dataset_sha256=fingerprint(rows),prompt_sha256=fingerprint(texts),status='running',
              max_new_tokens=max_new_tokens,batch_size=batch_size,decoding='greedy',enable_thinking=False,prompt_instruction=PROMPT,
              device='cpu',dtype='float32',torch_threads=torch.get_num_threads(),adapter=Path(adapter).name if adapter else None,
              versions={'torch':torch.__version__,'transformers':transformers.__version__},
              created_utc=datetime.now(timezone.utc).isoformat())
    save_json(output/'metadata.json',meta)
    total_start=time.perf_counter()
    predictions=[]
    with (output/'predictions.jsonl').open('w') as stream:
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            inputs=tokenizer(texts[start:start+batch_size],add_special_tokens=False,return_tensors='pt',padding=True)
            tic=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id,use_cache=True)
            seconds=time.perf_counter()-tic
            new=generated[:,inputs['input_ids'].shape[1]:]
            for i,row in enumerate(batch):
                tokens=new[i].tolist()
                eos=tokenizer.eos_token_id
                if eos in tokens: tokens=tokens[:tokens.index(eos)+1]
                prediction=dict(id=row['id'],raw_output=tokenizer.decode(tokens,skip_special_tokens=True),
                                output_tokens=len(tokens),input_tokens=int(inputs['attention_mask'][i].sum()),
                                batch_index=start//batch_size,batch_latency_seconds=seconds,
                                reached_token_cap=len(tokens)>=max_new_tokens)
                predictions.append(prediction)
                stream.write(json.dumps(prediction)+'\n')
                stream.flush()
            print(json.dumps({'phase':'inference','run':output.name,'done':min(start+batch_size,len(rows)),
                              'total':len(rows),'batch_seconds':round(seconds,2)}),flush=True)
    meta.update(status='complete',inference_seconds=time.perf_counter()-total_start,
                generated_tokens=sum(p['output_tokens'] for p in predictions),
                token_cap_fraction=sum(p['reached_token_cap'] for p in predictions)/len(predictions))
    save_json(output/'metadata.json',meta)
    summary,errors=evaluate(rows,predictions,meta,1000)
    save_json(output/'summary.json',summary)
    write_jsonl(output/'errors.jsonl',errors)
    (output/'report.md').write_text(markdown_report(summary))
    print(json.dumps({'phase':'completed','run':output.name,'metrics':summary['metrics'],'seconds':meta['inference_seconds']}),flush=True)
    del model,tokenizer
    gc.collect()


def main():
    import faulthandler
    faulthandler.dump_traceback_later(900, exit=True)
    p=argparse.ArgumentParser()
    p.add_argument('--model-local',required=True)
    p.add_argument('--study',default=str(Path(__file__).parent/'local-pilot'))
    p.add_argument('--scratch',required=True)
    p.add_argument('--mode',choices=['timing','baseline','train'],required=True)
    p.add_argument('--arm',choices=['B','C','E'])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    study=Path(args.study);scratch=Path(args.scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    plan=json.loads((study/'preregistration.json').read_text())
    all_rows=read_jsonl(study/'data.jsonl')
    validate(all_rows)
    if fingerprint(all_rows)!=plan['dataset_sha256']: raise ValueError('Frozen data changed')
    if args.mode=='timing':
        dev=[r for r in all_rows if r['split']=='dev' and r['group_id'].endswith('-0000')]
        infer(args.model_local,plan['revision'],dev,scratch/f'development-answer-only-b{args.batch_size}',batch_size=args.batch_size)
    elif args.mode=='baseline':
        infer(args.model_local,plan['revision'],[r for r in all_rows if r['split']=='test'],study/'A',batch_size=args.batch_size)
    else:
        if args.arm is None or args.seed not in plan['training_seeds']: raise ValueError('Use a registered arm and seed')
        from semantic_shift.training import train
        name=f'{args.arm}-seed{args.seed}'
        config=dict(model=str(Path(args.model_local).resolve()),revision=directory_digest(args.model_local),
                    condition=args.arm,seed=args.seed,device='cpu',max_length=plan['max_length'],
                    learning_rate=plan['learning_rate'],token_budget=plan['training_token_budget_per_arm'],
                    max_steps=plan['max_steps'],lora_rank=plan['lora_rank'],lora_alpha=plan['lora_alpha'],
                    target_modules=plan['target_modules'],chat_template_kwargs={'enable_thinking':False},prompt_instruction=PROMPT,answer_format='integer_only',dtype='float32')
        cfg_path=scratch/f'{name}.json';save_json(cfg_path,config)
        checkpoint=scratch/name
        start=time.perf_counter()
        print(json.dumps({'phase':'training-start','run':name}),flush=True)
        train(cfg_path,study/'data.jsonl',checkpoint)
        duration=time.perf_counter()-start
        manifest=json.loads((checkpoint/'training_manifest.json').read_text())
        manifest['config']['model']='<local snapshot of Qwen/Qwen3-0.6B>'
        manifest['hub_revision']=plan['revision']
        manifest['elapsed_seconds']=duration
        save_json(study/'training'/f'{name}-manifest.json',manifest)
        save_json(study/'training'/f'{name}-losses.json',json.loads((checkpoint/'losses.json').read_text()))
        print(json.dumps({'phase':'training-complete','run':name,'seconds':duration,'steps':manifest['steps']}),flush=True)
        gc.collect()
        infer(args.model_local,plan['revision'],[r for r in all_rows if r['split']=='test'],study/name,checkpoint,batch_size=args.batch_size)


if __name__=='__main__':main()
