#!/usr/bin/env python3
"""导演 Agent：把用户一句话创意扩写成 1-3 个分镜（shot list）。

输出的 JSON 同时是生成输入（wan_prompt）和评估输入（expected_subjects/camera/
transition —— 评估时豁免转场点 T4、静止镜头 T3 先验、T8/T5 主体跟踪词）。

用法: /opt/pytorch/bin/python3 director.py "一只柯基在雪地里追飞盘" --n-shots 3
"""
import argparse
import json
import sys

from common import ask_claude

DIRECTOR_PROMPT = """You are a film director agent for an AI text-to-video pipeline (Wan T2V).
Break the user's idea into exactly {n_shots} shots that together tell a coherent
mini-story. Per-shot video duration is about {dur} seconds.

User idea (may be in Chinese): "{idea}"

Requirements for each shot:
- "wan_prompt": an ENGLISH text-to-video prompt, 40-80 words, concrete and visual:
  subject appearance (keep identical wording across shots for the same subject —
  the video model has no memory between shots), action, environment, lighting,
  camera movement, film style. Single continuous shot, no cuts.
- "expected_subjects": list of short open-vocabulary detector phrases (e.g.
  "a corgi dog", "a red frisbee") for the subjects that MUST stay visible.
- "camera": one of "static" | "slow_pan" | "tracking" | "handheld" | "orbit" | "zoom".
- "motion_level": "low" | "medium" | "high"  (how much the content moves; a
  freeze-detector uses this as prior).
- "transition_to_next": "cut" | "fade" | "dissolve"  (last shot: null).

Respond ONLY with JSON:
{{"title": "<short English title>", "title_zh": "<短中文标题>",
  "style": "<overall visual style, English>",
  "shots": [{{"shot_id": 1, "wan_prompt": "...", "expected_subjects": ["..."],
             "camera": "...", "motion_level": "...", "duration_s": {dur},
             "transition_to_next": "cut|fade|dissolve|null"}}]}}"""


def direct(idea, n_shots=3, dur=5):
    parsed, raw = ask_claude(
        [{'text': DIRECTOR_PROMPT.format(idea=idea, n_shots=n_shots, dur=dur)}],
        max_tokens=2000)
    if not parsed or 'shots' not in parsed:
        raise RuntimeError(f'director parse fail: {raw[:300]}')
    parsed['idea'] = idea
    for i, s in enumerate(parsed['shots']):
        s['shot_id'] = i + 1
        if i == len(parsed['shots']) - 1:
            s['transition_to_next'] = None
    return parsed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('idea')
    ap.add_argument('--n-shots', type=int, default=3)
    ap.add_argument('--duration', type=float, default=5)
    a = ap.parse_args()
    print(json.dumps(direct(a.idea, a.n_shots, a.duration), ensure_ascii=False, indent=1))
