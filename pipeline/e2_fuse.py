#!/usr/bin/env python3
"""E2 第二步：把 probes/*.json 的原始信号融合成逐帧 anomaly score + Top-K 候选窗口。

融合规则（对应 video_eval.md 4.1 节公式）：
  每路信号 = robust z-score（相对本视频的 median/MAD）× 绝对下限门控。
  纯 z-score 会在正常视频上放大噪声（clean 视频里 z=4 可能只对应帧差 3.8 灰度级），
  所以每路信号必须同时超过"绝对物理下限"才计分。下限由 clean 校准视频的
  max × 安全系数初始化——这对应计划里"权重先用规则初始化，再用人工标注校准"。

  黑帧/冻结/硬切/主体缺失走独立的 hard 规则通道，不参与 z 归一化。

同时对照 ground_truth 评估：事件级 recall 与 temporal IoU。
"""
import json
import os

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
PROBES = os.path.join(BASE, "probes")
VIDEOS = os.path.join(BASE, "videos")

# 绝对下限：来自 clean.mp4 的信号最大值 × ~1.5 安全系数（首版规则，后续人工校准）
FLOORS = {"flicker": 1.5, "diff_d1": 8.0, "clip_dist_d1": 0.15, "flow_warp_residual": 5.0}
WEIGHTS = {"flicker": 1.0, "diff_d1": 0.5, "clip_dist_d1": 1.0, "flow_warp_residual": 1.5}


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def fuse(d):
    sig = d["signals"]
    n = d["n_frames"]
    score = np.zeros(n, np.float32)
    parts = {}
    for k, w in WEIGHTS.items():
        raw = np.asarray(sig[k], np.float32)
        z = np.clip(zscore(raw), 0, 8)          # 截顶，防单路信号淹没其他
        gate = (raw > FLOORS[k]).astype(np.float32)
        parts[k] = w * z * gate
        score += parts[k]
    # hard 规则通道
    lum = np.asarray(sig["luminance"], np.float32)
    black = (lum < 12).astype(np.float32) * 10
    d1 = np.asarray(sig["diff_d1"], np.float32)
    freeze = np.zeros(n, np.float32)
    freeze[1:] = (d1[1:] < 0.05) * 6            # 完全重复帧
    cut = np.zeros(n, np.float32)
    for c in d["scene_cuts_frames"]:
        if 0 <= c < n:
            cut[c] = 10
    areas = np.array([s["visible_area"] for s in d["subject"]], np.float32)
    med_area = np.median(areas[areas > 200]) if (areas > 200).any() else 1.0
    subj = np.zeros(n, np.float32)
    for i, s in enumerate(d["subject"]):
        if s["bbox"] is None:
            subj[i] = 6.0                        # 必需主体缺失
        elif s["visible_area"] / med_area < 0.45:
            subj[i] = 4.0                        # 可见面积骤降（疑似裁切）
        elif s["border_touch"]:
            subj[i] = 1.5                        # 边缘碰撞只是弱候选信号
    score += black + freeze + cut + subj
    return score, {"soft": parts, "black": black, "freeze": freeze, "cut": cut, "subject": subj}


def candidate_windows(score, fps, top_k=5, min_gap_s=0.75, half_w_s=0.6, thresh=2.5):
    s = score.copy()
    wins = []
    for _ in range(top_k):
        i = int(np.argmax(s))
        if s[i] < thresh:
            break
        wins.append({"peak_frame": i, "peak_s": round(i / fps, 2),
                     "score": round(float(s[i]), 2),
                     "start_s": round(max(0, i / fps - half_w_s), 2),
                     "end_s": round(i / fps + half_w_s, 2)})
        lo, hi = max(0, i - int(min_gap_s * fps)), min(len(s), i + int(min_gap_s * fps))
        s[lo:hi] = -1
    return wins


def interval_iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    gt_all = {g["video"]: g for g in json.load(open(os.path.join(VIDEOS, "ground_truth.json")))}
    report = []
    for name in sorted(gt_all):
        pj = os.path.join(PROBES, name.replace(".mp4", ".json"))
        if not os.path.exists(pj):
            continue
        d = json.load(open(pj))
        fps = d["fps"]
        score, _ = fuse(d)
        wins = candidate_windows(score, fps)
        d["anomaly_score"] = [round(float(v), 2) for v in score]
        d["candidate_windows"] = wins
        json.dump(d, open(pj, "w"), ensure_ascii=False)

        events = gt_all[name]["events"]
        hits = []
        for ev in events:
            best = 0.0
            covered = False
            for w in wins:
                iou = interval_iou(ev["start_s"], ev["end_s"], w["start_s"], w["end_s"])
                best = max(best, iou)
                if w["start_s"] <= (ev["start_s"] + ev["end_s"]) / 2 <= w["end_s"]:
                    covered = True
            hits.append({"type": ev["type"], "gt": [ev["start_s"], ev["end_s"]],
                         "covered": covered, "best_iou": round(best, 2)})
        report.append({"video": name, "n_windows": len(wins),
                       "windows": [(w["peak_s"], w["score"]) for w in wins],
                       "events": hits})
        rec = sum(h["covered"] for h in hits)
        print(f"{name:20s} windows={len(wins)} event_recall={rec}/{len(hits)} "
              f"peaks={[(w['peak_s'], w['score']) for w in wins]}")
        for h in hits:
            print(f"   GT {h['type']:15s} {h['gt']}  covered={h['covered']}  IoU={h['best_iou']}")
    json.dump(report, open(os.path.join(BASE, "results", "e2_fusion_eval.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
