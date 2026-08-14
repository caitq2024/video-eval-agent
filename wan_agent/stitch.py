#!/usr/bin/env python3
"""拼接：按导演的 transition_to_next 用 ffmpeg 拼成片。

- cut: concat filter（无过渡，硬切点 = 预期切换，评估时豁免 T4）
- fade / dissolve: xfade filter（0.5s 交叠）
返回 transitions 元数据：每个转场在成片时间轴上的 [start_s, end_s] 与类型，
评估管线用它豁免 T4/T1。
"""
import subprocess

from common import FFMPEG

FADE_DUR = 0.5
XFADE_NAME = {'fade': 'fade', 'dissolve': 'dissolve'}


def probe_duration(path):
    out = subprocess.run([FFMPEG, '-i', path], capture_output=True, text=True).stderr
    import re
    m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', out)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def stitch(clips, transitions, out_path, fps):
    """clips: 路径列表; transitions: len-1 个 'cut'|'fade'|'dissolve'。
    返回 [{'type', 'start_s', 'end_s'}] （成片时间轴）。"""
    assert len(transitions) == len(clips) - 1
    durs = [probe_duration(c) for c in clips]
    inputs = []
    for c in clips:
        inputs += ['-i', c]

    # 统一走 filter_complex。注意：concat/xfade 的中间输出会丢失 CFR 声明，
    # xfade 又强制要求恒定帧率 —— 每级输出后都要补一个 fps= 过滤器。
    pre = ''.join(f'[{i}:v]fps={fps}[v{i}];' for i in range(len(clips)))
    cur = 'v0'
    cur_end = durs[0]          # 当前累计输出时长
    meta = []
    parts = pre
    for i, tr in enumerate(transitions):
        nxt = f'v{i + 1}'
        outl = f'm{i}'
        if tr in XFADE_NAME:
            off = cur_end - FADE_DUR
            parts += (f'[{cur}][{nxt}]xfade=transition={XFADE_NAME[tr]}:'
                      f'duration={FADE_DUR}:offset={off:.3f},fps={fps}[{outl}];')
            meta.append({'type': tr, 'start_s': round(off, 2),
                         'end_s': round(off + FADE_DUR, 2)})
            cur_end = off + durs[i + 1]      # xfade 后总时长 = offset + 下一段全长
        else:  # cut
            parts += f'[{cur}][{nxt}]concat=n=2:v=1:a=0,fps={fps}[{outl}];'
            meta.append({'type': 'cut', 'start_s': round(cur_end, 2),
                         'end_s': round(cur_end, 2)})
            cur_end = cur_end + durs[i + 1]
        cur = outl
    graph = parts.rstrip(';')
    cmd = [FFMPEG, '-y'] + inputs + ['-filter_complex', graph, '-map', f'[{cur}]',
                                     '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
                                     '-pix_fmt', 'yuv420p', out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'ffmpeg stitch fail: {r.stderr[-800:]}')
    return meta
