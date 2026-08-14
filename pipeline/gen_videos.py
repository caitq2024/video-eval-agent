#!/usr/bin/env python3
"""生成带已知缺陷的合成测试视频 + ground truth。

场景设计：带纹理的移动背景（模拟缓慢镜头右移）+ 一个纹理圆形主体做平滑正弦运动。
镜头运动和主体运动都是"正常"信号源，用来考验探针的误报控制。

缺陷注入（每条视频只注入一种，便于归因）：
  temporal_jump     t=3.0s 处跳过 12 帧（主体瞬移）
  flicker           t=2.1 / 4.5 / 6.2s 各 1 帧亮度闪变
  freeze            t=5.0-5.75s 冻结（重复帧）
  hardcut           t=4.0-4.5s 意外切到另一场景再切回
  out_of_frame      t=3.0-5.0s 主体无理由漂出画面 60% 再回来
  black_frame       t=2.5s 2 帧全黑 + t=6.0s 1 帧强噪声
"""
import json
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "videos")
W, H, FPS, DUR = 640, 360, 16, 8.0
N = int(FPS * DUR)  # 128 帧

rng = np.random.RandomState(42)
# 预生成一张大背景纹理，镜头在其上平移（真实的全局运动）
BG_BIG = None


def make_bg():
    global BG_BIG
    big = rng.randint(40, 90, (H + 100, W + 400, 3)).astype(np.uint8)
    big = cv2.GaussianBlur(big, (31, 31), 0)
    # 加一些方块地标，让光流/特征有结构可依
    for _ in range(60):
        x, y = rng.randint(0, W + 340), rng.randint(0, H + 40)
        c = tuple(int(v) for v in rng.randint(60, 200, 3))
        cv2.rectangle(big, (x, y), (x + rng.randint(15, 60), y + rng.randint(15, 60)), c, -1)
    BG_BIG = big


def make_bg2():
    """hardcut 用的另一个场景"""
    big = rng.randint(150, 220, (H, W, 3)).astype(np.uint8)
    big = cv2.GaussianBlur(big, (15, 15), 0)
    for _ in range(30):
        x, y = rng.randint(0, W), rng.randint(0, H)
        cv2.circle(big, (x, y), rng.randint(10, 40), tuple(int(v) for v in rng.randint(0, 120, 3)), -1)
    return big


def subject_center(i, out_of_frame=False):
    """主体中心轨迹：平滑正弦。out_of_frame 时 t=3~5s 漂出画面"""
    t = i / FPS
    x = 120 + (W - 240) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 8.0 - np.pi / 2))
    y = H / 2 + 60 * np.sin(2 * np.pi * t / 4.0)
    if out_of_frame and 3.0 <= t <= 5.0:
        # 平滑地漂出右边界（圆心最多超出边界 ~0.6*半径*2）
        p = np.sin(np.pi * (t - 3.0) / 2.0)  # 0->1->0
        x = x + p * (W - x + 40)
    return int(x), int(y)


def render_frame(i, out_of_frame=False):
    # 镜头缓慢右移：每秒 20px
    ox = int(i / FPS * 20)
    frame = BG_BIG[50:50 + H, ox:ox + W].copy()
    cx, cy = subject_center(i, out_of_frame)
    r = 42
    # 纹理主体：圆 + 内部图案，保证被裁切时可检测
    cv2.circle(frame, (cx, cy), r, (30, 90, 200), -1)
    cv2.circle(frame, (cx, cy), r, (200, 220, 240), 3)
    cv2.circle(frame, (cx - 12, cy - 10), 8, (240, 240, 240), -1)
    cv2.circle(frame, (cx + 12, cy - 10), 8, (240, 240, 240), -1)
    cv2.ellipse(frame, (cx, cy + 12), (16, 8), 0, 0, 180, (240, 240, 240), 2)
    return frame


def write_video(name, frames, gt_events):
    path = os.path.join(OUT, f"{name}.mp4")
    tmp = path + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    # 转 H.264 保证 VLM/浏览器可读
    os.system(f"ffmpeg -y -loglevel error -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 20 {path}")
    os.remove(tmp)
    return {"video": f"{name}.mp4", "fps": FPS, "n_frames": len(frames), "events": gt_events}


def main():
    os.makedirs(OUT, exist_ok=True)
    make_bg()
    base = [render_frame(i) for i in range(N)]
    gt = []

    # 1. clean
    gt.append(write_video("clean", base, []))

    # 2. temporal_jump: t=3.0s 跳过 12 帧
    j0 = int(3.0 * FPS)
    frames = base[:j0] + base[j0 + 12:]
    gt.append(write_video("temporal_jump", frames,
                          [{"type": "temporal_jump", "start_s": round((j0 - 1) / FPS, 2),
                            "end_s": round((j0 + 1) / FPS, 2), "severity": 4}]))

    # 3. flicker: 3 个孤立亮闪帧
    frames = [f.copy() for f in base]
    ev = []
    for ts in (2.1, 4.5, 6.2):
        k = int(ts * FPS)
        frames[k] = cv2.convertScaleAbs(frames[k], alpha=1.9, beta=60)
        ev.append({"type": "flicker", "start_s": round(k / FPS, 2),
                   "end_s": round((k + 1) / FPS, 2), "severity": 3})
    gt.append(write_video("flicker", frames, ev))

    # 4. freeze: t=5.0s 起冻结 12 帧
    f0 = int(5.0 * FPS)
    frames = [f.copy() for f in base]
    for k in range(f0, f0 + 12):
        frames[k] = frames[f0].copy()
    gt.append(write_video("freeze", frames,
                          [{"type": "freeze", "start_s": round(f0 / FPS, 2),
                            "end_s": round((f0 + 12) / FPS, 2), "severity": 3}]))

    # 5. hardcut: t=4.0-4.5s 切到另一场景
    bg2 = make_bg2()
    c0, c1 = int(4.0 * FPS), int(4.5 * FPS)
    frames = [f.copy() for f in base]
    for k in range(c0, c1):
        f2 = bg2.copy()
        cv2.putText(f2, "WRONG SCENE", (180, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        frames[k] = f2
    gt.append(write_video("hardcut", frames,
                          [{"type": "unexpected_cut", "start_s": round(c0 / FPS, 2),
                            "end_s": round(c1 / FPS, 2), "severity": 5}]))

    # 6. out_of_frame
    frames = [render_frame(i, out_of_frame=True) for i in range(N)]
    gt.append(write_video("out_of_frame", frames,
                          [{"type": "subject_crop", "start_s": 3.4, "end_s": 4.6, "severity": 4}]))

    # 7. black_frame + 噪声帧
    frames = [f.copy() for f in base]
    b0 = int(2.5 * FPS)
    frames[b0] = np.zeros((H, W, 3), np.uint8)
    frames[b0 + 1] = np.zeros((H, W, 3), np.uint8)
    n0 = int(6.0 * FPS)
    frames[n0] = rng.randint(0, 255, (H, W, 3)).astype(np.uint8)
    gt.append(write_video("black_frame", frames,
                          [{"type": "black_frame", "start_s": round(b0 / FPS, 2),
                            "end_s": round((b0 + 2) / FPS, 2), "severity": 4},
                           {"type": "corrupt_frame", "start_s": round(n0 / FPS, 2),
                            "end_s": round((n0 + 1) / FPS, 2), "severity": 4}]))

    with open(os.path.join(OUT, "ground_truth.json"), "w") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print("done:", [g["video"] for g in gt])


if __name__ == "__main__":
    main()
