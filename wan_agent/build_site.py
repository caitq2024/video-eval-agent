#!/usr/bin/env python3
"""聚合 wan_outputs 下的 shots.json + eval.json → docs/data/data.json，
并把视频转码成 web 友好的 H.264（≤720p, crf 26, faststart）放进 docs/assets/videos/。

用法: /opt/pytorch/bin/python3 build_site.py [--skip-videos]
"""
import argparse
import json
import os
import subprocess

from common import EVAL_ROOT, FFMPEG, OUT_ROOT

REPO = os.path.join(EVAL_ROOT, 'repo')
DOCS = os.path.join(REPO, 'docs')
VID_DIR = os.path.join(DOCS, 'assets', 'videos')


def transcode(src, dst, max_h=480):
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return
    vf = f"scale=-2:'min({max_h},ih)'"
    r = subprocess.run([FFMPEG, '-y', '-i', src, '-vf', vf, '-c:v', 'libx264',
                        '-crf', '26', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', '-an', dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'transcode fail {src}: {r.stderr[-300:]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-videos', action='store_true')
    a = ap.parse_args()
    os.makedirs(VID_DIR, exist_ok=True)
    data = {'prompts': [], 'models': ['wan2.1', 'wan2.2'],
            'model_info': {
                'wan2.1': {'label': 'Wan2.1-T2V-1.3B', 'size': '832x480', 'fps': 16},
                'wan2.2': {'label': 'Wan2.2-TI2V-5B', 'size': '1280x704', 'fps': 24}}}
    for pid in sorted(os.listdir(OUT_ROOT)):
        if pid.startswith('demo_'):
            continue                     # 现场 demo 任务不进展示站
        sp = os.path.join(OUT_ROOT, pid, 'shots.json')
        if not os.path.exists(sp):
            continue
        meta = json.load(open(sp))
        entry = {'id': pid, 'idea': meta['idea'], 'title': meta.get('title'),
                 'title_zh': meta.get('title_zh'), 'style': meta.get('style'),
                 'director_wall_s': meta.get('director_wall_s'),
                 'shots': meta['shots'], 'generation': meta.get('generation', {}),
                 'films': meta.get('films', {}), 'evals': {}}
        for mk in ('wan2.1', 'wan2.2'):
            ep = os.path.join(OUT_ROOT, pid, mk, 'eval.json')
            if os.path.exists(ep):
                e = json.load(open(ep))
                e.pop('shots', None)              # 已在顶层
                entry['evals'][mk] = e
            if not a.skip_videos:
                for name in ['film.mp4'] + [f'shot{s["shot_id"]}.mp4'
                                            for s in meta['shots']]:
                    src = os.path.join(OUT_ROOT, pid, mk, name)
                    if os.path.exists(src) and os.path.getsize(src) > 0:
                        dst = os.path.join(VID_DIR, f'{pid}_{mk}_{name}')
                        transcode(src, dst)
        data['prompts'].append(entry)
    out = os.path.join(DOCS, 'data', 'data.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, 'w'), ensure_ascii=False)
    sz = sum(os.path.getsize(os.path.join(VID_DIR, f)) for f in os.listdir(VID_DIR)
             if f.endswith('.mp4')) / 1e6 if os.path.isdir(VID_DIR) else 0
    print(f'data.json written ({len(data["prompts"])} prompts); videos {sz:.0f} MB')


if __name__ == '__main__':
    main()
