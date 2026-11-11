#!/usr/bin/env python3
"""Launch a NEW bounded experiment, using the green artifact's CLI.
Dry-run by default. Does not patch production or rerun the resume probe.
Both arms use a NEW 3000-step linear schedule, NOT the 100k schedule.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact', required=True)
p.add_argument('--arm', choices=['baseline', 'p3'], required=True)
p.add_argument('--output', required=True, help='NEW empty/nonexistent directory')
p.add_argument('--center', required=True)
p.add_argument('--clean-root', required=True)
p.add_argument('--margin', required=True, type=float)
p.add_argument('--beta', required=True, type=float, help='0 for baseline; explicit positive candidate for P3')
p.add_argument('--execute', action='store_true')
a = p.parse_args()
if not math.isfinite(a.beta) or not math.isfinite(a.margin) or a.margin < 0:
    p.error('beta/margin must be finite; margin must be nonnegative')
if (a.arm == 'baseline' and a.beta != 0) or (a.arm == 'p3' and a.beta <= 0):
    p.error('baseline requires beta=0; p3 requires beta>0')
if int(os.environ.get('WORLD_SIZE', '1')) != 1:
    p.error('single process only; do not use torchrun')
center = Path(a.center).resolve()
clean = Path(a.clean_root).resolve()
out = Path(a.output).resolve()
if not center.is_file() or not clean.is_dir():
    p.error('center file and clean-root directory must exist')
if out.exists() and any(out.iterdir()):
    p.error('output must be empty: old artifacts will not be overwritten')
with zipfile.ZipFile(a.artifact) as z:
    cmd = json.loads(z.read('probe/command.json'))
    lock = json.loads(z.read('code_lock.json'))
for entry in lock['files'].values():
    f = Path(entry['path'])
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit(f'Code-lock mismatch: {f}; no training started')
script = Path(cmd[0]).resolve()
if not script.is_file():
    raise SystemExit(f'Missing reviewed entry point: {script}')
def option(k, v):
    if k in cmd:
        cmd[cmd.index(k) + 1] = str(v)
    else:
        cmd.extend([k, str(v)])
for k, v in {
    '--output_dir': out, '--model_name': 'pilot', '--max_steps': 3000,
    '--eval_steps': 1000, '--save_steps': 1000, '--logging_steps': 50,
    '--save_total_limit': 3, '--num_workers': 0,
}.items():
    option(k, v)
if '--resume_from_checkpoint' in cmd:
    raise SystemExit('Unexpected resume argument; this launcher starts fresh')
for k in ('--train_data', '--dev_data'):
    if not Path(cmd[cmd.index(k) + 1]).is_file():
        raise SystemExit(f'Missing {k} manifest')
env = os.environ.copy()
env.update(SIDECAR_REC_LAM='0', SIDECAR_SEP_LAM='0',
           SIDECAR_P3_BETA=str(a.beta), SIDECAR_P3_MARGIN=str(a.margin),
           SIDECAR_P3_CENTER=str(center), SIDECAR_CLEAN_ROOT=str(clean),
           SIDECAR_LOAD_CLEAN='1', SIDECAR_KEEP_FEATS='1',
           PYTHONHASHSEED='42', P3_CODE_ROOT=str(script.parent))
env['PYTHONPATH'] = str(script.parent) + os.pathsep + env.get('PYTHONPATH', '')
# Seed before model construction in each fresh process. Trainer retains its
# standard seed=42; this CLI does not expose a seed option.
bootstrap = ('import runpy,sys; from transformers import set_seed; '
             'set_seed(42); s=sys.argv[1]; sys.argv=sys.argv[1:]; '
             'sys.path.insert(0,__import__("os").path.dirname(s)); '
             'runpy.run_path(s,run_name="__main__")')
launch = [sys.executable, '-c', bootstrap, *cmd]
manifest = dict(arm=a.arm, command=cmd, seed=42, schedule_steps=3000,
    finetune_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    center_sha256=hashlib.sha256(center.read_bytes()).hexdigest(),
    environment={k:v for k,v in env.items() if k.startswith('SIDECAR_') or k=='P3_CODE_ROOT'},
    note='New bounded experiment; not a continuation of the 100k schedule. No automatic long-run approval.')
print(json.dumps(manifest, indent=2), flush=True)
if not a.execute:
    print('DRY RUN ONLY. Add --execute to start actual training.')
    raise SystemExit(0)
out.mkdir(parents=True, exist_ok=True)
(out / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
with (out / 'train.log').open('x') as log:
    result = subprocess.run(launch, env=env, stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
