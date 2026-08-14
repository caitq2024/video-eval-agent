#!/usr/bin/env python3
"""端到端编排：创意 idea → 导演分镜 → 多卡生成 → 拼接成片 → 元数据落盘。

用法:
  /opt/pytorch/bin/python3 run_pipeline.py --ideas ideas.json [--models wan2.1 wan2.2]
  /opt/pytorch/bin/python3 run_pipeline.py --idea "一只柯基在雪地里追飞盘" --pid p0

ideas.json: [{"id": "p1", "idea": "...", "n_shots": 3}, ...]
产物: wan_outputs/<pid>/shots.json + <model>/shot<k>.mp4 + <model>/film.mp4
"""
import argparse
import json
import os
import time

from common import MODELS, OUT_ROOT
from director import direct
from generate_clips import run_jobs
from stitch import stitch


def plan_prompt(pid, idea, n_shots, models, force=False):
    pdir = os.path.join(OUT_ROOT, pid)
    os.makedirs(pdir, exist_ok=True)
    meta_path = os.path.join(pdir, 'shots.json')
    if os.path.exists(meta_path) and not force:
        meta = json.load(open(meta_path))
        print(f'[director] {pid} 复用已有分镜: {meta["title"]}')
    else:
        t0 = time.time()
        meta = direct(idea, n_shots)
        meta['id'] = pid
        meta['director_wall_s'] = round(time.time() - t0, 1)
        json.dump(meta, open(meta_path, 'w'), ensure_ascii=False, indent=1)
        print(f'[director] {pid} "{meta["title"]}" {len(meta["shots"])} shots '
              f'({meta["director_wall_s"]}s)')
    jobs = []
    for mk in models:
        os.makedirs(os.path.join(pdir, mk), exist_ok=True)
        for s in meta['shots']:
            out = os.path.join(pdir, mk, f'shot{s["shot_id"]}.mp4')
            if os.path.exists(out) and not force:
                continue
            jobs.append({'model': mk, 'prompt': s['wan_prompt'],
                         'seed': 4200 + s['shot_id'],       # 同分镜跨模型同 seed
                         'out_path': out, 'duration_s': s['duration_s'],
                         'pid': pid, 'shot_id': s['shot_id']})
    return meta, jobs


def stitch_prompt(pid, meta, models):
    pdir = os.path.join(OUT_ROOT, pid)
    for mk in models:
        clips = [os.path.join(pdir, mk, f'shot{s["shot_id"]}.mp4') for s in meta['shots']]
        clips = [c for c in clips if os.path.exists(c)]
        film = os.path.join(pdir, mk, 'film.mp4')
        if len(clips) < 2:
            if len(clips) == 1:
                import shutil
                shutil.copy(clips[0], film)
                meta.setdefault('films', {})[mk] = {'transitions': []}
            continue
        trans = [s['transition_to_next'] or 'cut' for s in meta['shots'][:len(clips) - 1]]
        t0 = time.time()
        tmeta = stitch(clips, trans, film, MODELS[mk]['fps'])
        meta.setdefault('films', {})[mk] = {
            'transitions': tmeta, 'stitch_wall_s': round(time.time() - t0, 1)}
        print(f'[stitch] {pid}/{mk} -> film.mp4 transitions={tmeta}')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ideas')
    ap.add_argument('--idea')
    ap.add_argument('--pid', default='p0')
    ap.add_argument('--n-shots', type=int, default=3)
    ap.add_argument('--models', nargs='+', default=['wan2.1', 'wan2.2'])
    ap.add_argument('--gpus', type=int, nargs='+')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    ideas = (json.load(open(a.ideas)) if a.ideas
             else [{'id': a.pid, 'idea': a.idea, 'n_shots': a.n_shots}])

    all_jobs, metas = [], {}
    for it in ideas:
        meta, jobs = plan_prompt(it['id'], it['idea'], it.get('n_shots', a.n_shots),
                                 a.models, a.force)
        metas[it['id']] = meta
        all_jobs += jobs

    if all_jobs:
        results = run_jobs(all_jobs, a.gpus)
    else:
        results = []
        print('[gen] 所有 clip 已存在，跳过生成')

    by_pid = {}
    for r in results:
        by_pid.setdefault(r['pid'], []).append(r)
    for pid, meta in metas.items():
        gen = meta.setdefault('generation', {})
        for r in by_pid.get(pid, []):
            gen.setdefault(r['model'], {})[f'shot{r["shot_id"]}'] = {
                k: r[k] for k in ('ok', 'gen_wall_s', 'frames', 'size', 'fps', 'gpu',
                                  'seed', 'log_tail') if k in r}
        meta = stitch_prompt(pid, meta, a.models)
        json.dump(meta, open(os.path.join(OUT_ROOT, pid, 'shots.json'), 'w'),
                  ensure_ascii=False, indent=1)
    print('[done]', list(metas))


if __name__ == '__main__':
    main()
