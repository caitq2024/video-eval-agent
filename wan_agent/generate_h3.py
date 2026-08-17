#!/usr/bin/env python3
"""MiniMax-H3 生成客户端：把分镜 h3_prompt 提交到本地 SGLang 服务逐镜生成 5s，
下载到 wan_outputs/<pid>/minimax-h3/，并拼接成片（复用 stitch，24fps）。

用法: /opt/pytorch/bin/python3 generate_h3.py [--pids p1 p2 ...] [--base http://localhost:30010]
      [--concurrency 2] [--force]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import OUT_ROOT
from run_pipeline import stitch_prompt

MK = 'minimax-h3'


def gen_one(base, prompt, out_path, seed, dur=5, timeout_s=3600):
    t0 = time.time()
    r = requests.post(f'{base}/v1/videos', json={
        'task': 't2va', 'prompt': prompt, 'conditions': [],
        'target': {'short_edge': 768, 'aspect_ratio': '16:9',
                   'duration_seconds': dur},
        'seed': seed}, timeout=120)
    r.raise_for_status()
    vid = r.json()['id']
    while True:
        st = requests.get(f'{base}/v1/videos/{vid}', timeout=60).json()
        status = st.get('status')
        if status in ('completed', 'succeeded', 'success'):
            break
        if status in ('failed', 'error', 'cancelled'):
            return {'ok': False, 'gen_wall_s': round(time.time() - t0, 1),
                    'log_tail': json.dumps(st)[:500]}
        if time.time() - t0 > timeout_s:
            return {'ok': False, 'gen_wall_s': round(time.time() - t0, 1),
                    'log_tail': f'timeout, last status={status}'}
        time.sleep(5)
    data = requests.get(f'{base}/v1/videos/{vid}/content', timeout=300).content
    open(out_path, 'wb').write(data)
    return {'ok': len(data) > 10000, 'gen_wall_s': round(time.time() - t0, 1),
            'bytes': len(data), 'log_tail': ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', nargs='+', default=[f'p{i}' for i in range(1, 11)])
    ap.add_argument('--base', default='http://localhost:30010')
    ap.add_argument('--concurrency', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    jobs = []
    metas = {}
    for pid in a.pids:
        mp = os.path.join(OUT_ROOT, pid, 'shots.json')
        meta = json.load(open(mp))
        metas[pid] = meta
        os.makedirs(os.path.join(OUT_ROOT, pid, MK), exist_ok=True)
        for s in meta['shots']:
            out = os.path.join(OUT_ROOT, pid, MK, f'shot{s["shot_id"]}.mp4')
            if os.path.exists(out) and os.path.getsize(out) > 10000 and not a.force:
                continue
            if 'h3_prompt' not in s:
                raise RuntimeError(f'{pid} shot{s["shot_id"]} 缺 h3_prompt，先跑 h3_promptify.py')
            jobs.append((pid, s['shot_id'], s['h3_prompt'],
                         out, 4200 + s['shot_id'], s.get('duration_s', 5)))
    print(f'[h3] {len(jobs)} jobs, concurrency={a.concurrency}')

    def run(j):
        pid, sid, prompt, out, seed, dur = j
        print(f'[h3] start {pid}/shot{sid}')
        try:
            res = gen_one(a.base, prompt, out, seed, int(dur))
        except Exception as e:
            res = {'ok': False, 'gen_wall_s': 0, 'log_tail': f'{type(e).__name__}: {e}'[:300]}
        res.update({'seed': seed, 'fps': 24, 'size': '1344*768'})
        print(f'[h3] done  {pid}/shot{sid} ok={res["ok"]} {res["gen_wall_s"]}s')
        return pid, sid, res

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results = list(pool.map(run, jobs))

    for pid, sid, res in results:
        metas[pid].setdefault('generation', {}).setdefault(MK, {})[f'shot{sid}'] = res
    for pid, meta in metas.items():
        clips = [os.path.join(OUT_ROOT, pid, MK, f'shot{s["shot_id"]}.mp4')
                 for s in meta['shots']]
        if all(os.path.exists(c) and os.path.getsize(c) > 10000 for c in clips):
            meta = stitch_prompt(pid, meta, [MK])
        json.dump(meta, open(os.path.join(OUT_ROOT, pid, 'shots.json'), 'w'),
                  ensure_ascii=False, indent=1)
    print('[h3] all done')


if __name__ == '__main__':
    main()
