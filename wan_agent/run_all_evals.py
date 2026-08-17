#!/usr/bin/env python3
"""批量评估所有成片并汇总耗时。用法:
  /opt/pytorch/bin/python3 run_all_evals.py [--device cuda:0] [--force]
"""
import argparse
import json
import os
import time

from common import OUT_ROOT
from evaluate import evaluate

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('HF_HOME', '/home/ec2-user/hf_cache')
    os.environ.setdefault('TORCH_HOME', '/home/ec2-user/torch_cache')
    t0 = time.time()
    rows = []
    for pid in sorted(os.listdir(OUT_ROOT)):
        if pid.startswith('demo_'):
            continue                     # 客户现场 demo 任务不进正式批量
        for mk in ('wan2.1', 'wan2.2', 'minimax-h3'):
            film = os.path.join(OUT_ROOT, pid, mk, 'film.mp4')
            ej = os.path.join(OUT_ROOT, pid, mk, 'eval.json')
            if not os.path.exists(film) or os.path.getsize(film) == 0:
                continue
            if os.path.exists(ej) and not a.force:
                e = json.load(open(ej))
                print(f'[skip] {pid}/{mk} already evaluated')
            else:
                try:
                    e = evaluate(pid, mk, a.device)
                except Exception as ex:
                    print(f'[FAIL] {pid}/{mk}: {ex}')
                    continue
            rows.append((pid, mk, e['scores']['total'], len(e['findings']),
                         e['timing']['total_s'], e['timing'].get('vlm_calls', 0)))
    print(f'\n=== 汇总（总耗时 {time.time() - t0:.0f}s）===')
    print(f'{"pid":5s} {"model":8s} {"score":>6s} {"finds":>6s} {"eval_s":>7s} {"vlm":>4s}')
    for r in rows:
        print(f'{r[0]:5s} {r[1]:8s} {r[2]:6.1f} {r[3]:6d} {r[4]:7.1f} {r[5]:4d}')
