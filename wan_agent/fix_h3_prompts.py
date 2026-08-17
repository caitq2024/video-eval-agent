#!/usr/bin/env python3
"""按 MiniMax 官方 h3-prompt-writing 规范修正 p5/p6/p7/p10 的 h3_prompt。

每段 clip 是独立 T2VA 生成，官方规定单镜 prompt 必须以 [Shot 1] 开头且不带时间戳；
运镜句式统一为 motion type + with small/large amplitude + at slow/fast speed（pan 带方向）；
non_diegetic_music 去掉抽象情绪词。原文备份到 h3_prompt_orig。
"""
import json
import os

from common import OUT_ROOT

FIXES = {
    ('p5', 2): [('[Shot 2] ', '[Shot 1] '),
                ('The camera moves as a tracking shot following the subject laterally at a fast speed, matching the athletes\' motion.',
                 'The camera follows the athletes laterally in a tracking shot with large amplitude at fast speed.')],
    ('p5', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pushes in at slow speed on the point of contact.',
                 'The camera pushes in with small amplitude at slow speed toward the point of contact.')],
    ('p6', 2): [('[Shot 2] ', '[Shot 1] '),
                ('The camera pans slowly with small amplitude, gradually revealing more of the storefront and its details.',
                 'The camera pans right with small amplitude at slow speed, gradually revealing more of the storefront and its details.')],
    ('p6', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pushes in at slow speed, gradually embracing the whole storefront.',
                 'The camera pushes in with small amplitude at slow speed toward the whole storefront.')],
    ('p7', 1): [('The camera performs a downward tracking shot following the ball at a fast, wide amplitude, staying locked on the falling basketball in cinematic slow motion.',
                 'The camera follows the falling ball downward in a tracking shot with large amplitude at fast speed, staying locked on it in cinematic slow motion.'),
                ('with a faint low-frequency pulse building anticipation beneath it.',
                 'with a faint low-frequency pulse that gradually increases in volume beneath it.')],
    ('p7', 2): [('[Shot 2] ', '[Shot 1] '),
                ('The camera performs a vertical tracking shot following the ball\'s bounces at a moderate amplitude and steady speed, holding it in cinematic slow motion.',
                 'The camera follows the ball\'s bounces vertically in a tracking shot, holding it in cinematic slow motion.')],
    ('p7', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pushes in at slow speed on the settled ball in cinematic slow motion.',
                 'The camera pushes in with small amplitude at slow speed toward the settled ball in cinematic slow motion.')],
    ('p10', 1): [('The camera pushes in at slow speed on his sleeping face over the full duration.',
                  'The camera pushes in with small amplitude at slow speed toward his sleeping face over the full duration.')],
    ('p10', 2): [('[Shot 2] ', '[Shot 1] '),
                 ('A tracking shot following the subject stays low to the ground with the cat throughout.',
                  'The camera follows the cat in a tracking shot, staying low to the ground throughout.'),
                 ('quick playful staccato rhythm', 'quick staccato rhythm')],
    ('p10', 3): [('[Shot 3] ', '[Shot 1] '),
                 ('The camera pans slowly with small amplitude from the clerk to the escaping cat.',
                  'The camera pans right with small amplitude at slow speed from the clerk to the escaping cat.'),
                 ('A brief startled string stab', 'A brief sharp string stab')],
}

for pid in ['p5', 'p6', 'p7', 'p10']:
    mp = os.path.join(OUT_ROOT, pid, 'shots.json')
    meta = json.load(open(mp))
    for s in meta['shots']:
        key = (pid, s['shot_id'])
        fixes = FIXES.get(key, [])
        p = s['h3_prompt']
        orig = p
        for old, new in fixes:
            if old not in p:
                raise RuntimeError(f'{key}: 找不到待替换文本: {old[:60]}')
            p = p.replace(old, new)
        if p != orig:
            s.setdefault('h3_prompt_orig', orig)
            s['h3_prompt'] = p
            print(f'{pid}/shot{s["shot_id"]}: {len(fixes)} 处修正')
        else:
            print(f'{pid}/shot{s["shot_id"]}: 无需修正')
        assert s['h3_prompt'].startswith('integrated_multimodal_description: [Shot 1]'), key
    json.dump(meta, open(mp, 'w'), ensure_ascii=False, indent=1)
print('done')
