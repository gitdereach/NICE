"""
Reproduce Benchmark 1 (elasto-plastic Drucker-Prager material) of
Masi & Einav (2024), "Neural integration for constitutive equations using small data", CMAME 420, 116698.

Runs the original notebook `benchmarks/[NICE] Benchmark-1.ipynb` headless, without modifying it:
  --mode pretrained : skip training and load the authors' weights from benchmarks/saved/[state]NICE_benchmark1
  --mode train      : train from scratch (optionally capped with --max-epochs); weights are written to
                      reproduction/output/ instead of overwriting benchmarks/saved/

Outputs (reproduction/output/): executed notebook, figures (PNG) and a JSON with the stress MAE values.

Usage (from the repository root):
    python reproduction/run_benchmark1.py --mode pretrained
    python reproduction/run_benchmark1.py --mode train --max-epochs 5000
"""
import argparse
import base64
import json
import os
import re
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, 'benchmarks')
NB = os.path.join(BENCH, '[NICE] Benchmark-1.ipynb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pretrained', 'train'], default='pretrained')
    ap.add_argument('--max-epochs', type=int, default=None, help='cap on training epochs (train mode)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reproduction', 'output'))
    args = ap.parse_args()
    out = os.path.join(args.out, args.mode)
    os.makedirs(out, exist_ok=True)

    nb = nbformat.read(NB, as_version=4)
    cells = nb.cells
    for c in cells:
        if c.cell_type != 'code':
            continue
        src = c.source
        if src.lstrip().startswith('# Training loop'):
            if args.mode == 'pretrained':
                c.source = ("NICE_network.load_state_dict(torch.load('./saved/[state]NICE_benchmark1'))\n"
                            "print('loaded authors weights')")
        if 'Nepochs = 1000000' in src and args.max_epochs:
            c.source = src.replace('Nepochs = 1000000', f'Nepochs = {args.max_epochs + 1}')
        if "checkpoint_path = './checkpoints/checkpoint.pt'" in src:
            c.source = c.source.replace("'./checkpoints/checkpoint.pt'", repr(os.path.join(out, 'checkpoint.pt')))
        if "PATH = './saved/[state]NICE_benchmark1'" in src:
            c.source = src.replace("'./saved/[state]NICE_benchmark1'", repr(os.path.join(out, 'state_NICE_benchmark1.pt')))
        if "params_path = './saved/[params]NICE_benchmark1'" in src:
            c.source = src.replace("'./saved/[params]NICE_benchmark1'", repr(os.path.join(out, 'params_NICE_benchmark1')))

    t0 = time.time()
    NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': BENCH}}).execute()
    elapsed = time.time() - t0
    nbformat.write(nb, os.path.join(out, 'Benchmark-1_executed.ipynb'))

    res, k = {'mode': args.mode, 'elapsed_s': round(elapsed, 1)}, 0
    for i, c in enumerate(cells):
        cell_txt = ''.join(o.get('text', '') for o in c.get('outputs', []))
        m = re.findall(r'MAE stress:\s+tensor\(([0-9.eE+-]+)', cell_txt)
        if m:
            res['MAE_stress_trainval' if 'MAE_stress_trainval' not in res else 'MAE_stress_test'] = float(m[0])
        for o in c.get('outputs', []):
            txt = o.get('text', '')
            ep = re.findall(r'Epoch: (\d+) \| training loss: ([0-9.eE+-]+) \| validation loss: ([0-9.eE+-]+)', txt)
            if ep:
                res['last_epoch'], res['train_loss'], res['val_loss'] = int(ep[-1][0]), float(ep[-1][1]), float(ep[-1][2])
            if 'data' in o and 'image/png' in o['data']:
                k += 1
                with open(os.path.join(out, f'fig{k}_cell{i}.png'), 'wb') as f:
                    f.write(base64.b64decode(o['data']['image/png']))
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
