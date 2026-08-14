#!/usr/bin/env python3
"""E7：裁决层升级对比 —— Nova Lite vs Claude Sonnet 4.5 vs Claude + sample-specific rubric。

聚焦 hybrid 方案中"必须由 VLM 出结论"的五个裁决案例：
  tax_swap_A   主体 4.5s 起变紫   → 期望 identity_change（Nova 判对）
  tax_swap_B   主体 2.8s 起变绿   → 期望 identity_change（Nova 漏判——T8 的那半个缺口）
  tax_crop_A   3.4-4.6s 异常出界  → 期望 out_of_frame
  normal_exit  prompt 要求结尾出画 → 期望 intentional（误报对照）
  closeup      prompt 要求特写贴边 → 期望 intentional（误报对照）

rubric 先行（VideoArgus 思路）：看到视频之前先由 judge 从 prompt 生成 3-5 条硬标准，
随后的裁决 prompt 附带该 rubric，防止被输出内容"带着走"。
"""
import json
import os
import time

import numpy as np

import vlm_common as V

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NOVA = 'us.amazon.nova-2-lite-v1:0'
CLAUDE = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0'

GEN_TAX = (V.GEN_PROMPT + ' A green square floats gently near the left edge and remains '
           'visible for the whole video.')

CASES = [  # (name, intended_prompt, window(s), expected verdict)
    ('tax_swap_A', GEN_TAX, (4.5, 8.0), 'identity_change'),
    ('tax_swap_B', GEN_TAX, (2.8, 8.0), 'identity_change'),
    ('tax_crop_A', V.GEN_PROMPT, (3.4, 4.6), 'out_of_frame'),
    ('normal_exit',
     'A single continuous shot: a textured dark background with scattered colored '
     'rectangles, camera slowly panning right. An orange smiley-face ball moves smoothly, '
     'then EXITS THE FRAME to the right during the final two seconds and does not return.',
     (7.5, 8.0), 'intentional'),
    ('closeup',
     'A single continuous EXTREME CLOSE-UP shot of a large orange smiley-face ball filling '
     'most of the frame; its edges may extend beyond the frame borders by design.',
     (0.0, 8.0), 'intentional'),
]


def ask(model, blocks, max_tokens=600):
    import re
    r = V.client().converse(modelId=model,
                            messages=[{'role': 'user', 'content': blocks}],
                            inferenceConfig={'maxTokens': max_tokens, 'temperature': 0.0})
    text = r['output']['message']['content'][0]['text']
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None, text
    try:
        return json.loads(m.group(0)), text
    except json.JSONDecodeError:
        return None, text


def make_rubric(model, prompt_text):
    p = (f'You are designing evaluation criteria for an AI-generated video BEFORE seeing it '
         f'(to avoid being biased by the output). The generation prompt was:\n"{prompt_text}"\n'
         f'Write 3-5 hard pass/fail criteria focused on subject visibility, identity '
         f'consistency (color/appearance must stay constant), and framing. '
         f'Respond ONLY with JSON: {{"criteria": ["...", ...]}}')
    parsed, _ = ask(model, [{'text': p}])
    return (parsed or {}).get('criteria', [])


def adjudicate(model, name, prompt_text, win, rubric=None):
    frames, fps = V.read_frames(os.path.join(BASE, 'videos', name + '.mp4'))
    n = len(frames)
    c = int((win[0] + win[1]) / 2 * fps)
    lo = max(0, min(n - 8, c - 4))
    sheet = V.contact_sheet([(frames[k], k / fps) for k in range(lo, lo + 8)],
                            cols=4, tile_w=300)
    rub = ''
    if rubric:
        rub = '\nPre-registered evaluation criteria (written before seeing the video):\n' + \
              '\n'.join(f'- {x}' for x in rubric) + '\n'
    p = (f'You are inspecting an AI-generated video. Intended prompt:\n"{prompt_text}"\n{rub}\n'
         f'A tracking tool flagged t={win[0]}-{win[1]}s: the main subject (orange ball) is '
         f'missing, much smaller, or touching the frame border there. Below are 8 consecutive '
         f'frames from that region.\nClassify what actually happened:\n'
         f'- "out_of_frame": ball left frame / got cropped with no semantic reason\n'
         f'- "identity_change": ball still present but its appearance (color/face) changed '
         f'vs the prompt\n'
         f'- "intentional": composition consistent with the prompt (scripted exit, close-up)\n'
         f'- "other_defect"\n'
         f'Respond ONLY with JSON: {{"verdict": "out_of_frame|identity_change|intentional|'
         f'other_defect", "reason": "<short>", "confidence": 0.0-1.0}}')
    parsed, raw = ask(model, [{'text': p}, V.img_block(sheet)])
    return (parsed or {}).get('verdict', 'PARSE_FAIL'), (parsed or {}).get('reason', raw[:100])


def main():
    arms = [('nova', NOVA, False), ('claude', CLAUDE, False), ('claude+rubric', CLAUDE, True)]
    results = {}
    rubrics = {}
    for name, ptxt, win, expect in CASES:
        for arm, model, use_rubric in arms:
            rub = None
            if use_rubric:
                if ptxt not in rubrics:
                    rubrics[ptxt] = make_rubric(model, ptxt)
                    time.sleep(0.3)
                rub = rubrics[ptxt]
            v, reason = adjudicate(model, name, ptxt, win, rub)
            ok = (v == expect) or (expect == 'intentional' and v == 'intentional')
            results.setdefault(arm, []).append(ok)
            print(f'{name:12s} {arm:14s} -> {v:16s} (expect {expect:16s}) '
                  f'{"OK" if ok else "MISS"} | {reason[:60]}')
            time.sleep(0.4)
    print('\n=== 汇总（5 个裁决案例）===')
    for arm, oks in results.items():
        print(f'{arm:14s} {sum(oks)}/{len(oks)}')
    json.dump({'results': {k: v for k, v in results.items()}, 'rubrics': list(rubrics.values())},
              open(os.path.join(BASE, 'results', 'e7_claude_verdict.json'), 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
