"""Download exactly the public checkpoint used in the study; no inference API charges."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
path=Path(a.output)
path.mkdir(parents=True,exist_ok=True)
snapshot_download('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',
                  local_dir=path,cache_dir=path/'.download-cache',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md','LICENSE'],max_workers=2)
print(path.resolve())
