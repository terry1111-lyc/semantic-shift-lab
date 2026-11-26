"""Sequential reproduction of the float32 pilot. No paid compute is provisioned."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--model-local',required=True)
p.add_argument('--scratch',required=True)
p.add_argument('--study',required=True,help='Copy the frozen data and preregistration into a new results directory')
a=p.parse_args()
script=Path(__file__).with_name('run_local_study.py')
base=[sys.executable,str(script),'--model-local',a.model_local,'--scratch',a.scratch,'--study',a.study,'--batch-size','8']
jobs=[('A',['--mode','baseline'])]+[(f'{arm}-seed{seed}',['--mode','train','--arm',arm,'--seed',str(seed)]) for seed in (17,29) for arm in 'BCE']
for name,args in jobs:
    folder=Path(a.study)/name
    if folder.exists():
        raise SystemExit(f'{folder} already exists. Inspect and archive previous output before starting a new sweep.')
for name,args in jobs:
    print(json.dumps({'starting':name}),flush=True)
    subprocess.run(base+args,check=True)
