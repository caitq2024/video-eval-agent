#!/usr/bin/env python3
"""E3b：hybrid 路由 —— detector 硬规则直接出结论，VLM 只做语义裁决。

E3 第一轮发现：CaC 式窗口选择 9/9 覆盖缺陷，但把窗口丢给 VLM 泛泛地问
"有没有缺陷"，Nova 在 temporal_jump / freeze / out_of_frame 窗口上返回空。
瓶颈不在采样，在窗口内判定。

hybrid 策略（对应计划 4.1"规则信号 + VLM 裁决"）：
  - 黑帧 / 冻结 / 硬切 / 闪烁：探针已有像素级证据（lum<12、diff≈0、PySceneDetect、
    亮度尖峰），直接由 detector 生成 finding，时间窗取信号连续段 → 时间定位也更准；
  - 主体缺失/骤降：交给 VLM + intended prompt 裁决"有意出画还是异常裁切"；
  - 软信号（flow warp residual / CLIP jump）：VLM 定向提问——明确告诉它该窗口
    哪个信号触发、逐格检查主体位置是否连续。
"""
import json
import os
import time

import numpy as np

import vlm_common as V
from e0_vlm_sampling import eval_findings
from e3_routing import best_iou, FLOORS

BASE = os.path.join(os.path.dirname(__file__), "..")
VIDEOS = os.path.join(BASE, "videos")
PROBES = os.path.join(BASE, "probes")
RESULTS = os.path.join(BASE, "results")


def runs_where(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def detector_findings(d, fps):
    """从探针信号直接生成硬规则 findings（含精确时间窗）"""
    sig = d["signals"]
    n = d["n_frames"]
    lum = np.asarray(sig["luminance"], np.float32)
    d1 = np.asarray(sig["diff_d1"], np.float32)
    flick = np.asarray(sig["flicker"], np.float32)
    out = []
    for i, j in runs_where(lum < 12):
        out.append({"type": "black_frame", "start_s": round(i / fps, 2),
                    "end_s": round(j / fps, 2), "severity": 4,
                    "evidence": f"mean luminance <12 for {j - i} frame(s)", "confidence": 0.99})
    frz = np.zeros(n, bool)
    frz[1:] = d1[1:] < 0.05
    for i, j in runs_where(frz):
        if j - i >= 3:  # ≥3 帧完全重复
            out.append({"type": "freeze", "start_s": round((i - 1) / fps, 2),
                        "end_s": round(j / fps, 2), "severity": 3,
                        "evidence": f"{j - i} consecutive duplicated frames (diff≈0)",
                        "confidence": 0.99})
    for c in d["scene_cuts_frames"]:
        out.append({"type": "unexpected_cut", "start_s": round(c / fps, 2),
                    "end_s": round((c + 1) / fps, 2), "severity": 5,
                    "evidence": "shot boundary detected in a single-shot task", "confidence": 0.95})
    for i, j in runs_where(flick > 1.5):
        # 排除黑帧引起的亮度尖峰（黑帧另行报告）
        if lum[i:j].min() >= 12 and (i == 0 or lum[i - 1] >= 12):
            out.append({"type": "flicker", "start_s": round(i / fps, 2),
                        "end_s": round(j / fps, 2), "severity": 3,
                        "evidence": f"luminance spike {flick[i:j].max():.1f} vs neighbors",
                        "confidence": 0.9})
    return out


def subject_vlm(d, frames, fps, prompt_text):
    """主体缺失/骤降段 → VLM 语义裁决（同 E4）"""
    subj = d["subject"]
    areas = np.array([s["visible_area"] for s in subj], np.float32)
    med = np.median(areas[areas > 200]) if (areas > 200).any() else 1.0
    miss = np.array([s["bbox"] is None or s["visible_area"] / med < 0.45 for s in subj])
    out = []
    n = len(frames)
    for i, j in runs_where(miss):
        if j - i < 2:
            continue
        c = (i + j) // 2
        lo = max(0, min(n - 8, c - 4))
        sheet = V.contact_sheet([(frames[k], k / fps) for k in range(lo, lo + 8)],
                                cols=4, tile_w=300)
        p = (f"You are inspecting an AI-generated video. Intended prompt:\n\"{prompt_text}\"\n\n"
             f"A tracking tool reports the main subject (orange ball) is missing or much "
             f"smaller during t={i / fps:.2f}-{j / fps:.2f}s. Below are 8 consecutive frames "
             f"from that region.\nIs this an INTENTIONAL composition per the prompt (scripted "
             f"exit, close-up) or a GENERATION DEFECT (subject out of frame / cropped with no "
             f"semantic reason)?\nRespond ONLY with JSON: "
             f'{{"defect": true|false, "reason": "<short>", "confidence": 0.0-1.0}}')
        parsed, _ = V.ask_nova([{"text": p}, V.img_block(sheet)])
        time.sleep(0.3)
        if (parsed or {}).get("defect"):
            out.append({"type": "subject_out_of_frame", "start_s": round(i / fps, 2),
                        "end_s": round(j / fps, 2), "severity": 4,
                        "evidence": (parsed or {}).get("reason", ""), "confidence": 0.9})
    return out, len(runs_where(miss))


def soft_window_vlm(d, frames, fps, covered_s):
    """软信号窗口（未被硬规则覆盖的）→ VLM 定向提问"""
    out = []
    calls = 0
    n = len(frames)
    for w in d.get("candidate_windows", [])[:2]:
        pk = w["peak_frame"]
        t = pk / fps
        if any(abs(t - c) < 0.7 for c in covered_s):  # 已被硬规则解释
            continue
        fired = [k for k, fl in FLOORS.items() if d["signals"][k][pk] > fl]
        if not fired:
            continue
        lo = max(0, min(n - 8, pk - 4))
        sheet = V.contact_sheet([(frames[k], k / fps) for k in range(lo, lo + 8)],
                                cols=4, tile_w=300)
        p = (f"You are inspecting an AI-generated video. Intended prompt:\n\"{V.GEN_PROMPT}\"\n\n"
             f"Low-level detectors ({', '.join(fired)}) flagged a temporal anomaly peaking at "
             f"t={t:.2f}s. Below are 8 CONSECUTIVE frames ({1 / fps:.3f}s apart) around it.\n"
             f"Check tile by tile: (1) is the orange ball visible in EVERY tile? "
             f"(2) does its position move SMOOTHLY between consecutive tiles, or does it "
             f"teleport / jump much farther between one pair of tiles than the others? "
             f"(3) any sudden lighting or scene change?\n"
             f"A large position jump between adjacent tiles here means dropped/skipped frames "
             f"(temporal_jump).\n{V.DEFECT_SCHEMA}")
        parsed, _ = V.ask_nova([{"text": p}, V.img_block(sheet)], max_tokens=900)
        time.sleep(0.3)
        calls += 1
        out += (parsed or {}).get("findings", [])
    return out, calls


def run():
    gt_all = json.load(open(os.path.join(VIDEOS, "ground_truth.json")))
    records = []
    for g in gt_all:
        name = g["video"]
        frames, fps = V.read_frames(os.path.join(VIDEOS, name))
        d = json.load(open(os.path.join(PROBES, name.replace(".mp4", ".json"))))
        prompt_text = g.get("intended_prompt", V.GEN_PROMPT)

        hard = detector_findings(d, fps)
        subj_f, _ = subject_vlm(d, frames, fps, prompt_text)
        covered = [(f["start_s"] + f["end_s"]) / 2 for f in hard + subj_f]
        soft_f, soft_calls = soft_window_vlm(d, frames, fps, covered)
        findings = hard + subj_f + soft_f

        hits, fp = eval_findings(findings, g["events"])
        ious = [round(best_iou(findings, ev), 2) for ev in g["events"]]
        records.append({"video": name, "router": "hybrid", "hits": hits, "false_pos": fp,
                        "ious": ious, "findings": findings})
        print(f"{name:18s} recall={sum(hits)}/{len(hits)} fp={fp} iou={ious} "
              f"types={[f['type'] for f in findings]}")
    json.dump(records, open(os.path.join(RESULTS, "e3b_hybrid.json"), "w"),
              ensure_ascii=False, indent=2)
    h = sum(sum(r["hits"]) for r in records)
    t = sum(len(r["hits"]) for r in records)
    fp = sum(r["false_pos"] for r in records)
    ious = [i for r in records for i in r["ious"]]
    print(f"\n=== E3b hybrid 汇总 === recall={h}/{t} ({h / max(t, 1):.0%}) false_pos={fp} "
          f"mean_IoU={np.mean(ious) if ious else 0:.2f}")


if __name__ == "__main__":
    run()
