#!/usr/bin/env python3
"""对真实长视频（演示/动效类）做全帧扫描 → anomaly timeline → Top-K 候选窗。

与合成实验的差异（内容先验不同）：
  - 演示视频大量静止画面是正常的 → 冻结不再作为缺陷信号；
  - 有意的场景切换/转场很多 → PySceneDetect 的 cut 位置对跳变类信号降权 0.3，
    "不在剪辑点上的帧间跳变"才是最可疑的（疑似渲染毛刺/编码故障）；
  - 没有固定主体 → 跳过主体跟踪，出界/裁切类问题交给窗口 VLM 顺带检查。

单次解码双时间轴：全帧率算亮度/闪烁/黑帧/帧差；每 3 帧存 320px 小图给 CLIP+RAFT。
用法：python3 e5_real_scan.py <video_path> [out_json]
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

cv2.setNumThreads(4)
torch.set_num_threads(5)

BASE = os.path.join(os.path.dirname(__file__), "..")
FLOW_SIZE = (320, 184)
SUB_EVERY = 3          # 25fps / 3 ≈ 8.3fps 子采样给 CLIP/RAFT
TOP_K = 6
MIN_GAP_S = 3.0

# 绝对下限（灰度级/余弦距离），低于此的波动不参与报警
FLOORS = {"flicker": 3.0, "warp": 8.0, "clip": 0.20}


def decode(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lum, small_grays, sub_frames, sub_idx = [], [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        small_grays.append(g)
        lum.append(float(g.mean()))
        if i % SUB_EVERY == 0:
            sub_frames.append(cv2.resize(f, FLOW_SIZE))
            sub_idx.append(i)
        i += 1
    cap.release()
    return fps, lum, small_grays, sub_frames, sub_idx


def pixel_signals(lum, grays):
    n = len(grays)
    d1 = np.zeros(n, np.float32)
    for i in range(1, n):
        d1[i] = np.abs(grays[i] - grays[i - 1]).mean()
    flick = np.zeros(n, np.float32)
    la = np.asarray(lum, np.float32)
    flick[1:-1] = np.abs(la[1:-1] - (la[:-2] + la[2:]) / 2)
    black = la < 10
    return d1, flick, black


def clip_signal(sub_frames):
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sub_frames), 64):
            batch = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in sub_frames[i:i + 64]])
            feats.append(model.encode_image(batch))
    feats = torch.cat(feats)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    dist = np.zeros(len(sub_frames), np.float32)
    dist[1:] = (1 - (feats[1:] * feats[:-1]).sum(-1)).numpy()
    return dist


def flow_signal(sub_frames):
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    model = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
    h, w = FLOW_SIZE[1], FLOW_SIZE[0]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

    def prep(f):
        t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1
        return t.unsqueeze(0)

    warp = np.zeros(len(sub_frames), np.float32)
    with torch.no_grad():
        for i in range(len(sub_frames) - 1):
            flow = model(prep(sub_frames[i]), prep(sub_frames[i + 1]))[-1][0].numpy()
            mx, my = gx + flow[0], gy + flow[1]
            warped = cv2.remap(sub_frames[i + 1].astype(np.float32), mx, my,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            warp[i + 1] = np.abs(warped - sub_frames[i].astype(np.float32)).mean()
    return warp


def scene_cuts(path):
    from scenedetect import detect, ContentDetector
    scenes = detect(path, ContentDetector(threshold=27.0), show_progress=False)
    return [s.get_frames() for s, _ in scenes[1:]]


def zpos(x):
    x = np.asarray(x, np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), 0, 8)


def fuse(n, fps, d1, flick, black, clip_d, warp, sub_idx, cuts):
    # 子采样信号回填到全帧率时间轴
    clip_full = np.zeros(n, np.float32)
    warp_full = np.zeros(n, np.float32)
    for k, i in enumerate(sub_idx):
        j = sub_idx[k + 1] if k + 1 < len(sub_idx) else n
        clip_full[i:j] = clip_d[k]
        warp_full[i:j] = warp[k]
    # 剪辑点 ±3 帧内属于"有意切换"，跳变信号降权
    cut_w = np.ones(n, np.float32)
    for c in cuts:
        cut_w[max(0, c - 3):min(n, c + 4)] = 0.3
    score = (1.5 * zpos(warp_full) * (np.asarray([warp_full > FLOORS["warp"]])[0]) * cut_w
             + 1.0 * zpos(clip_full) * (clip_full > FLOORS["clip"]) * cut_w
             + 1.2 * zpos(flick) * (flick > FLOORS["flicker"])
             + black.astype(np.float32) * 10)
    return score


def windows(score, fps, top_k=TOP_K, thresh=3.0):
    s = score.copy()
    out = []
    for _ in range(top_k):
        i = int(np.argmax(s))
        if s[i] < thresh:
            break
        out.append({"peak_frame": i, "peak_s": round(i / fps, 2),
                    "score": round(float(s[i]), 2)})
        lo, hi = max(0, i - int(MIN_GAP_S * fps)), min(len(s), i + int(MIN_GAP_S * fps))
        s[lo:hi] = -1
    return out


def main():
    path = sys.argv[1]
    name = os.path.basename(path)
    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        BASE, "probes", "real_" + name.replace(".mp4", ".json"))
    t0 = time.time()
    fps, lum, grays, sub_frames, sub_idx = decode(path)
    n = len(grays)
    t1 = time.time()
    d1, flick, black = pixel_signals(lum, grays)
    del grays
    t2 = time.time()
    clip_d = clip_signal(sub_frames)
    t3 = time.time()
    warp = flow_signal(sub_frames)
    t4 = time.time()
    cuts = scene_cuts(path)
    t5 = time.time()
    score = fuse(n, fps, d1, flick, black, clip_d, warp, sub_idx, cuts)
    wins = windows(score, fps)
    out = {"video": name, "path": path, "fps": fps, "n_frames": n,
           "duration_s": round(n / fps, 1),
           "timing_s": {"decode": round(t1 - t0, 1), "pixel": round(t2 - t1, 1),
                        "clip": round(t3 - t2, 1), "raft": round(t4 - t3, 1),
                        "scenedetect": round(t5 - t4, 1)},
           "n_cuts": len(cuts), "cuts_s": [round(c / fps, 2) for c in cuts],
           "signals_full": {"flicker": np.round(flick, 2).tolist(),
                            "diff_d1": np.round(d1, 2).tolist(),
                            "luminance": np.round(np.asarray(lum), 1).tolist()},
           "signals_sub": {"idx": sub_idx, "clip_dist": np.round(clip_d, 4).tolist(),
                           "warp_residual": np.round(warp, 2).tolist()},
           "anomaly_score": np.round(score, 2).tolist(),
           "candidate_windows": wins}
    json.dump(out, open(out_json, "w"))
    print(f"{name}: {n} frames, {len(cuts)} cuts, windows="
          f"{[(w['peak_s'], w['score']) for w in wins]}, "
          f"total {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
