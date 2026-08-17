#!/usr/bin/env python3
"""把现有导演分镜转写成 MiniMax-H3 T2VA 结构化 prompt（官方 h3-prompt-writing 规范）。

设计：复用 p1-p10 已有分镜（与 Wan 版同分镜、同 seed 语义 → 公平对比基座），
每个 shot 产出三字段结构 prompt 存入 shots.json 的 shots[k]['h3_prompt']。
逐镜独立生成 5s（与 Wan 管线对齐，转场仍由 ffmpeg 拼接并产出豁免元数据）。

用法: /opt/pytorch/bin/python3 h3_promptify.py [pid ...]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from common import OUT_ROOT, ask_claude

CAMERA_MAP = {
    'static': 'a static shot',
    'slow_pan': 'the camera pans slowly with small amplitude',
    'tracking': 'a tracking shot following the subject',
    'handheld': 'a handheld shot that shakes slightly',
    'orbit': 'an arc shot circling the subject at slow speed',
    'zoom': 'the camera pushes in at slow speed',
}

GUIDE = """You are rewriting a storyboard shot into MiniMax-H3 T2VA structured prompt
format (official h3-prompt-writing spec). Output EXACTLY three fields as plain text:

integrated_multimodal_description: [Shot 1] <style>, <framing>, ... single continuous
5-second shot. Start with overall style (e.g. "Live-action, cinematic" / "3D CG") and
initial composition, then subject appearance & position, scene & key props, actions &
reactions along the timeline, and any synchronized on-screen sounds. Camera movement
must be written as natural in-sentence English (motion type + amplitude + speed, omit
medium/normal). Do NOT add extra [Shot N] cuts — this is ONE continuous shot. On-screen
text (signs/banners) goes in double quotes verbatim. No dialogue (no <d> tags needed).

overall_soundscape: 1-3 English sentences, one paragraph, summarizing ambient sound,
physical action sounds, non-verbal vocals for THIS shot (wind, footsteps, fabric,
impacts...). No music here.

non_diegetic_music: 1-2 English sentences describing background score audible only to
the audience: instruments, tempo, rhythm, dynamics. NO abstract emotion words. Write
N/A if score would not fit.

Rules: keep the subject appearance wording IDENTICAL across shots of the same story
(the model has no memory between independently generated shots); keep all key visual
elements from the source prompt; enrich audio fields plausibly from the scene."""


def promptify(pid):
    mp = os.path.join(OUT_ROOT, pid, 'shots.json')
    meta = json.load(open(mp))
    shots_txt = json.dumps([{k: s[k] for k in
                             ('shot_id', 'wan_prompt', 'camera', 'motion_level',
                              'expected_subjects')} for s in meta['shots']],
                           ensure_ascii=False)
    cam_hints = {s['shot_id']: CAMERA_MAP.get(s.get('camera'), '')
                 for s in meta['shots']}
    p = (f'{GUIDE}\n\nStory idea (Chinese): "{meta["idea"]}"\n'
         f'Storyboard shots (source prompts written for Wan T2V):\n{shots_txt}\n'
         f'Camera phrasing hints per shot: {json.dumps(cam_hints)}\n\n'
         f'Rewrite EACH shot into the 3-field H3 format (each is an independent '
         f'5-second single-shot generation). Respond ONLY with JSON:\n'
         f'{{"shots": [{{"shot_id": 1, "h3_prompt": "integrated_multimodal_description: '
         f'[Shot 1] ...\\noverall_soundscape: ...\\nnon_diegetic_music: ..."}}]}}')
    parsed, raw = ask_claude([{'text': p}], max_tokens=4000)
    if not parsed or 'shots' not in parsed:
        raise RuntimeError(f'{pid} promptify parse fail: {raw[:200]}')
    got = {x['shot_id']: x['h3_prompt'] for x in parsed['shots']}
    for s in meta['shots']:
        s['h3_prompt'] = got[s['shot_id']]
    json.dump(meta, open(mp, 'w'), ensure_ascii=False, indent=1)
    return pid, [len(got[s['shot_id']]) for s in meta['shots']]


if __name__ == '__main__':
    pids = sys.argv[1:] or [f'p{i}' for i in range(1, 11)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        for pid, lens in pool.map(promptify, pids):
            print(f'{pid}: h3_prompt lens={lens}')
