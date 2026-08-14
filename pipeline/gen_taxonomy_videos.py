#!/usr/bin/env python3
"""拦截率测试集：每类缺陷 2 个变体（A/B 不同时间点/参数），共 10 类。

复用 gen_videos.py 的场景（纹理背景 + 镜头右移 + 橙色笑脸球正弦运动），
新增一个常驻次要物体（绿色方块，缓慢上下浮动）用于 object_vanish 类。

类型（对齐调研中的常见缺陷分类）：
  T1 temporal_jump    跳帧/瞬移           A: 3.0s 跳 12 帧   B: 5.5s 跳 8 帧
  T2 flicker          单帧亮度闪烁        A: 2.1/4.5/6.2s    B: 1.4/6.8s
  T3 freeze           冻结/重复帧         A: 5.0s 0.75s      B: 2.0s 1.0s
  T4 unexpected_cut   意外场景切换        A: 4.0-4.5s        B: 1.5-1.9s
  T5 subject_crop     主体异常出界        A: 3.4-4.6s        B: 5.5-6.7s（出下边界）
  T6 black_corrupt    黑帧/坏帧           A: 2.5s 黑×2+6.0s 噪  B: 4.0s 黑×3
  T7 deformation      主体变形            A: 3.5-4.1s 压扁    B: 6.0-6.6s 拉长
  T8 identity_swap    主体身份突变        A: 4.5s 起变紫色    B: 2.8s 起变绿脸
  T9 object_vanish    物体凭空消失        A: 方块 5.0s 消失0.5s后回  B: 3.2s 永久消失
  T10 semantic_mismatch 语义不符          A/B: clean 视频 + 两个错误 prompt（不生成新视频）
"""
import json
import os

import cv2
import numpy as np

import gen_videos as G

W, H, FPS, N = G.W, G.H, G.FPS, G.N
OUT = G.OUT


def draw_subject(frame, cx, cy, r=42, color=(30, 90, 200), ax_ratio=1.0):
    """ax_ratio!=1 → 变形（椭圆）"""
    rx, ry = int(r * ax_ratio), int(r / ax_ratio)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (200, 220, 240), 3)
    ex, ey = int(12 * ax_ratio), int(10 / ax_ratio)
    cv2.circle(frame, (cx - ex, cy - ey), 8, (240, 240, 240), -1)
    cv2.circle(frame, (cx + ex, cy - ey), 8, (240, 240, 240), -1)
    cv2.ellipse(frame, (cx, cy + int(12 / ax_ratio)), (int(16 * ax_ratio), 8),
                0, 0, 180, (240, 240, 240), 2)


def render(i, subj_color=(30, 90, 200), ax_ratio=1.0, subj_visible=True,
           square_visible=True, oof=False):
    ox = int(i / FPS * 20)
    frame = G.BG_BIG[50:50 + H, ox:ox + W].copy()
    t = i / FPS
    # 次要物体：绿色方块，左侧缓慢浮动
    if square_visible:
        sy = int(80 + 25 * np.sin(2 * np.pi * t / 6.0))
        cv2.rectangle(frame, (60, sy), (110, sy + 50), (60, 200, 60), -1)
        cv2.rectangle(frame, (60, sy), (110, sy + 50), (230, 240, 230), 2)
    if subj_visible:
        cx, cy = G.subject_center(i)
        if oof and 5.5 <= t <= 6.7:  # B 变体：漂出下边界
            p = np.sin(np.pi * (t - 5.5) / 1.2)
            cy = int(cy + p * (H - cy + 40))
        draw_subject(frame, cx, cy, color=subj_color, ax_ratio=ax_ratio)
    return frame


def base_frames(**kw):
    return [render(i, **kw) for i in range(N)]


def main():
    G.make_bg()
    gt = []

    def add(name, frames, events):
        gt.append(G.write_video(name, frames, events))

    base = base_frames()

    # T1 temporal_jump
    j = int(3.0 * FPS)
    add("tax_jump_A", base[:j] + base[j + 12:],
        [{"type": "temporal_jump", "start_s": round((j - 1) / FPS, 2),
          "end_s": round((j + 1) / FPS, 2), "severity": 4}])
    j = int(5.5 * FPS)
    add("tax_jump_B", base[:j] + base[j + 8:],
        [{"type": "temporal_jump", "start_s": round((j - 1) / FPS, 2),
          "end_s": round((j + 1) / FPS, 2), "severity": 4}])

    # T2 flicker
    for tag, tss in (("A", (2.1, 4.5, 6.2)), ("B", (1.4, 6.8))):
        fr = [f.copy() for f in base]
        ev = []
        for ts in tss:
            k = int(ts * FPS)
            fr[k] = cv2.convertScaleAbs(fr[k], alpha=1.9, beta=60)
            ev.append({"type": "flicker", "start_s": round(k / FPS, 2),
                       "end_s": round((k + 1) / FPS, 2), "severity": 3})
        add(f"tax_flicker_{tag}", fr, ev)

    # T3 freeze
    for tag, (t0, dur) in (("A", (5.0, 12)), ("B", (2.0, 16))):
        f0 = int(t0 * FPS)
        fr = [f.copy() for f in base]
        for k in range(f0, f0 + dur):
            fr[k] = fr[f0].copy()
        add(f"tax_freeze_{tag}", fr,
            [{"type": "freeze", "start_s": round(f0 / FPS, 2),
              "end_s": round((f0 + dur) / FPS, 2), "severity": 3}])

    # T4 unexpected_cut
    bg2 = G.make_bg2()
    for tag, (a, b) in (("A", (4.0, 4.5)), ("B", (1.5, 1.9))):
        c0, c1 = int(a * FPS), int(b * FPS)
        fr = [f.copy() for f in base]
        for k in range(c0, c1):
            f2 = bg2.copy()
            cv2.putText(f2, "WRONG SCENE", (180, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            fr[k] = f2
        add(f"tax_cut_{tag}", fr,
            [{"type": "unexpected_cut", "start_s": round(c0 / FPS, 2),
              "end_s": round(c1 / FPS, 2), "severity": 5}])

    # T5 subject_crop：A 出右边界（同旧实验），B 出下边界
    fr = [G.render_frame(i, out_of_frame=True) for i in range(N)]
    add("tax_crop_A", fr, [{"type": "subject_crop", "start_s": 3.4, "end_s": 4.6,
                            "severity": 4}])
    fr = base_frames(oof=True)
    add("tax_crop_B", fr, [{"type": "subject_crop", "start_s": 5.7, "end_s": 6.5,
                            "severity": 4}])

    # T6 black/corrupt
    fr = [f.copy() for f in base]
    b0 = int(2.5 * FPS)
    fr[b0] = np.zeros((H, W, 3), np.uint8)
    fr[b0 + 1] = np.zeros((H, W, 3), np.uint8)
    n0 = int(6.0 * FPS)
    fr[n0] = np.random.RandomState(1).randint(0, 255, (H, W, 3)).astype(np.uint8)
    add("tax_black_A", fr,
        [{"type": "black_frame", "start_s": round(b0 / FPS, 2),
          "end_s": round((b0 + 2) / FPS, 2), "severity": 4},
         {"type": "corrupt_frame", "start_s": round(n0 / FPS, 2),
          "end_s": round((n0 + 1) / FPS, 2), "severity": 4}])
    fr = [f.copy() for f in base]
    b0 = int(4.0 * FPS)
    for k in range(b0, b0 + 3):
        fr[k] = np.zeros((H, W, 3), np.uint8)
    add("tax_black_B", fr,
        [{"type": "black_frame", "start_s": round(b0 / FPS, 2),
          "end_s": round((b0 + 3) / FPS, 2), "severity": 4}])

    # T7 deformation：短时间轴比突变（压扁/拉长）
    for tag, (t0, t1, ar) in (("A", (3.5, 4.1, 1.7)), ("B", (6.0, 6.6, 0.55))):
        fr = []
        for i in range(N):
            t = i / FPS
            fr.append(render(i, ax_ratio=ar if t0 <= t <= t1 else 1.0))
        add(f"tax_deform_{tag}", fr,
            [{"type": "deformation", "start_s": t0, "end_s": t1, "severity": 4}])

    # T8 identity_swap：主体颜色永久突变
    for tag, (ts, col) in (("A", (4.5, (200, 60, 160))), ("B", (2.8, (60, 200, 60)))):
        k0 = int(ts * FPS)
        fr = [render(i, subj_color=col if i >= k0 else (30, 90, 200)) for i in range(N)]
        add(f"tax_swap_{tag}", fr,
            [{"type": "identity_swap", "start_s": round(k0 / FPS, 2),
              "end_s": round((k0 + 2) / FPS, 2), "severity": 4}])

    # T9 object_vanish：绿方块消失（A 短暂 0.5s，B 永久）
    v0 = int(5.0 * FPS)
    fr = [render(i, square_visible=not (v0 <= i < v0 + 8)) for i in range(N)]
    add("tax_vanish_A", fr,
        [{"type": "object_vanish", "start_s": round(v0 / FPS, 2),
          "end_s": round((v0 + 8) / FPS, 2), "severity": 3}])
    v0 = int(3.2 * FPS)
    fr = [render(i, square_visible=i < v0) for i in range(N)]
    add("tax_vanish_B", fr,
        [{"type": "object_vanish", "start_s": round(v0 / FPS, 2),
          "end_s": round((v0 + 2) / FPS, 2), "severity": 3}])

    # 基准 clean（带绿方块的新场景，供校准/对照）
    add("tax_clean", base, [])

    json.dump(gt, open(os.path.join(OUT, "taxonomy_ground_truth.json"), "w"),
              ensure_ascii=False, indent=2)
    print("generated:", [g["video"] for g in gt])


if __name__ == "__main__":
    main()
