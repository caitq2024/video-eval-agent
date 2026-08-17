#!/usr/bin/env python3
"""第二批：按 MiniMax 官方 h3-prompt-writing 规范修正 p1-p4/p8/p9 的 h3_prompt。
规则同 fix_h3_prompts.py：单镜独立生成以 [Shot 1] 开头；运镜写成
motion type + with amplitude + at speed 的自然句（pan 带方向）；
non_diegetic_music 去抽象情绪词。原文备份 h3_prompt_orig。
"""
import json
import os

from common import OUT_ROOT

FIXES = {
    ('p1', 1): [('A handheld shot that shakes slightly follows behind the soldier as he advances forward through the debris.',
                 'The camera follows behind the soldier in a tracking shot, shaking slightly, as he advances forward through the debris.'),
                ('with sparse muffled percussive hits that build tension gradually.',
                 'with sparse muffled percussive hits that build gradually in volume.')],
    ('p1', 2): [('[Shot 2] ', '[Shot 1] '),
                ('A tracking shot following the subject moves quickly alongside him from the side, matching his fast forward movement.',
                 'The camera follows him from the side in a tracking shot with large amplitude at fast speed.')],
    ('p1', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pushes in at slow speed toward his backlit silhouette.',
                 'The camera pushes in with small amplitude at slow speed toward his backlit silhouette.')],
    ('p2', 1): [('The camera pans slowly with small amplitude toward the dog.',
                 'The camera pans right with small amplitude at slow speed toward the dog.')],
    ('p2', 2): [('[Shot 2] ', '[Shot 1] '),
                ('A tracking shot following the subject moves alongside the corgi\'s motion from the side.',
                 'The camera follows the corgi from the side in a tracking shot.'),
                ('An uplifting acoustic guitar and light strings drive a moderate rolling tempo',
                 'An acoustic guitar and light strings drive a moderate rolling tempo')],
    ('p2', 3): [('[Shot 3] ', '[Shot 1] '),
                ('A static shot captures the triumphant moment from a low angle.',
                 'The camera holds a static shot from a low angle.'),
                ('A soaring orchestral swell of strings with a bright cymbal shimmer hits at the moment of the catch, slow majestic tempo rising to a full dynamic peak then settling gently.',
                 'An orchestral swell of strings with a bright cymbal shimmer hits at the moment of the catch, at a slow tempo rising to a full dynamic peak then settling gently.')],
    ('p3', 1): [('Filmed as a static shot.',
                 'The camera holds a static shot throughout.')],
    ('p3', 2): [('[Shot 2] ', '[Shot 1] '),
                ('The camera performs an arc shot circling the subject at slow speed.',
                 'The camera moves in an arc shot around the bottle at slow speed.')],
    ('p3', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pans slowly with small amplitude.',
                 'The camera pans left with small amplitude at slow speed.')],
    ('p4', 1): [],
    ('p4', 2): [('[Shot 2] ', '[Shot 1] '),
                ('The camera moves as a tracking shot following the subject, gliding alongside at a slow, smooth pace.',
                 'The camera follows the subject in a tracking shot, gliding alongside at slow speed.')],
    ('p4', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera is a handheld shot that shakes slightly, drifting with subtle motion to emphasize an immersive mood.',
                 'The camera shakes slightly in a handheld shot, drifting with subtle motion.')],
    ('p8', 1): [('An arc shot circling the subject at slow speed gradually reveals the surrounding park,',
                 'The camera moves in an arc shot around the subject at slow speed, gradually revealing the surrounding park,')],
    ('p8', 2): [('[Shot 2] ', '[Shot 1] '),
                ('An arc shot circling the subject at slow speed moves gently closer, synchronized with',
                 'The camera moves in an arc shot around the subject at slow speed, drawing gently closer, synchronized with')],
    ('p8', 3): [('[Shot 3] ', '[Shot 1] '),
                ('An arc shot circling the subject at slow speed orbits and gently rises for a tranquil final view, accompanied by',
                 'The camera moves in an arc shot around the subject at slow speed and rises gently, accompanied by')],
    ('p9', 1): [('The camera pans slowly with small amplitude around her, revealing the reflection.',
                 'The camera moves in an arc shot around her with small amplitude at slow speed, revealing the reflection.')],
    ('p9', 2): [('[Shot 2] ', '[Shot 1] '),
                ('A tracking shot follows the subject smoothly through the movement.',
                 'The camera follows her in a tracking shot smoothly through the movement.'),
                ('An elegant downtempo electronic track at around 100 BPM,',
                 'A downtempo electronic track at around 100 BPM,')],
    ('p9', 3): [('[Shot 3] ', '[Shot 1] '),
                ('The camera pushes in at slow speed onto her face and pose.',
                 'The camera pushes in with small amplitude at slow speed toward her face.'),
                ('A confident downtempo electronic track at around 95 BPM,',
                 'A downtempo electronic track at around 95 BPM,')],
}

for pid in ['p1', 'p2', 'p3', 'p4', 'p8', 'p9']:
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
