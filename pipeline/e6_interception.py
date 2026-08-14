#!/usr/bin/env python3
"""E6：10 类常见缺陷的拦截率测试（evaluation-first：先定义错误分类，再测拦截）。

每类 2 个变体（A/B），hybrid pipeline 拦截判定：
  存在与 GT 窗口时间重叠（±0.75s）且类型可接受的 finding → 拦截成功。

pipeline 组成：
  1) detector 硬规则：黑帧/冻结/硬切/闪烁（e3b 复用）+ 新增主体形变（bbox 纵横比突变）
  2) 主体缺失/骤降段 → VLM 语义裁决（扩展类型：出界/身份突变/其他）
  3) 软信号 Top-K 窗口 → VLM 定向提问（扩展 schema 含 deformation/identity_change/object_vanish）
  4) 语义一致性：8 帧均匀采样 + intended prompt → 对齐度 1-5（T10 专用，
     稀疏采样对全局语义判断是够用的——坏帧问题不适用于这一层）
"""
import json
import os
import time

import numpy as np

import vlm_common as V
from e3b_hybrid import detector_findings, runs_where
from e3_routing import FLOORS
import e2_fuse

BASE = os.path.join(os.path.dirname(__file__), "..")
VIDEOS = os.path.join(BASE, "videos")
PROBES = os.path.join(BASE, "probes")
RESULTS = os.path.join(BASE, "results")

TOL_S = 0.75

# 每类 GT 类型 → 可接受的预测类型
ALIAS = {
    "temporal_jump": {"temporal_jump", "freeze", "unexpected_cut", "other"},
    "flicker": {"flicker", "black_frame", "other"},
    "freeze": {"freeze", "temporal_jump", "other"},
    "unexpected_cut": {"unexpected_cut", "temporal_jump", "corrupt_frame", "other"},
    "subject_crop": {"subject_out_of_frame", "other"},
    "black_frame": {"black_frame", "flicker", "corrupt_frame"},
    "corrupt_frame": {"corrupt_frame", "black_frame", "flicker", "unexpected_cut", "other"},
    "deformation": {"deformation", "other"},
    "identity_swap": {"identity_change", "deformation", "other"},
    "object_vanish": {"object_vanish", "temporal_jump", "other"},
}

TAX_SCHEMA = V.DEFECT_SCHEMA.replace(
    "temporal_jump|flicker|freeze|unexpected_cut|subject_out_of_frame|black_frame|corrupt_frame|other",
    "temporal_jump|flicker|freeze|unexpected_cut|subject_out_of_frame|black_frame|"
    "corrupt_frame|deformation|identity_change|object_vanish|other")

GEN_PROMPT_TAX = (V.GEN_PROMPT + " A green square floats gently near the left edge and "
                  "remains visible for the whole video.")


def deformation_findings(d, fps):
    """bbox 纵横比突变（主体完整可见时）→ deformation"""
    subj = d["subject"]
    areas = np.array([s["visible_area"] for s in subj], np.float32)
    med_area = np.median(areas[areas > 200]) if (areas > 200).any() else 1.0
    asp = np.ones(len(subj), np.float32)
    valid = np.zeros(len(subj), bool)
    for i, s in enumerate(subj):
        if s["bbox"] and not s["border_touch"] and s["visible_area"] / med_area > 0.6:
            x0, y0, x1, y1 = s["bbox"]
            if y1 > y0:
                asp[i] = (x1 - x0 + 1) / (y1 - y0 + 1)
                valid[i] = True
    la = np.abs(np.log(asp / (np.median(asp[valid]) if valid.any() else 1.0)))
    out = []
    for i, j in runs_where((la > 0.3) & valid):
        if j - i >= 2:
            out.append({"type": "deformation", "start_s": round(i / fps, 2),
                        "end_s": round(j / fps, 2), "severity": 4,
                        "evidence": f"subject bbox aspect ratio deviates {la[i:j].max():.2f} "
                                    f"(log) from median for {j - i} frames",
                        "confidence": 0.9})
    return out


def subject_vlm_tax(d, frames, fps):
    """主体缺失/骤降段 → VLM 裁决（含身份突变判定）"""
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
        p = (f"You are inspecting an AI-generated video. Intended prompt:\n"
             f"\"{GEN_PROMPT_TAX}\"\n\n"
             f"A tracker matched to the ORANGE ball reports it missing or much smaller "
             f"during t={i / fps:.2f}-{j / fps:.2f}s. Below are 8 consecutive frames.\n"
             f"Classify what actually happened:\n"
             f'- "out_of_frame": ball left the frame / got cropped with no semantic reason\n'
             f'- "identity_change": the ball is still there but its appearance (color/face) '
             f"changed vs the prompt\n"
             f'- "intentional": composition consistent with the prompt\n'
             f'- "other_defect": something else wrong\n'
             f'Respond ONLY with JSON: {{"verdict": "out_of_frame|identity_change|'
             f'intentional|other_defect", "reason": "<short>", "confidence": 0.0-1.0}}')
        parsed, _ = V.ask_nova([{"text": p}, V.img_block(sheet)])
        time.sleep(0.3)
        v = (parsed or {}).get("verdict", "other_defect")
        if v != "intentional":
            tmap = {"out_of_frame": "subject_out_of_frame",
                    "identity_change": "identity_change"}
            out.append({"type": tmap.get(v, "other"), "start_s": round(i / fps, 2),
                        "end_s": round(j / fps, 2), "severity": 4,
                        "evidence": (parsed or {}).get("reason", ""), "confidence": 0.9})
    return out


def soft_window_vlm_tax(d, frames, fps, covered_s):
    out, n = [], len(frames)
    for w in d.get("candidate_windows", [])[:3]:
        pk = w["peak_frame"]
        t = pk / fps
        if any(abs(t - c) < 0.7 for c in covered_s):
            continue
        fired = [k for k, fl in FLOORS.items() if d["signals"][k][pk] > fl]
        if not fired:
            continue
        lo = max(0, min(n - 8, pk - 4))
        sheet = V.contact_sheet([(frames[k], k / fps) for k in range(lo, lo + 8)],
                                cols=4, tile_w=300)
        p = (f"You are inspecting an AI-generated video. Intended prompt:\n"
             f"\"{GEN_PROMPT_TAX}\"\n\n"
             f"Low-level detectors ({', '.join(fired)}) flagged an anomaly peaking at "
             f"t={t:.2f}s. Below are 8 CONSECUTIVE frames ({1 / fps:.3f}s apart).\n"
             f"Check tile by tile: (1) is the orange ball visible and does it keep the SAME "
             f"appearance (color, face) in every tile? (2) does its position move smoothly, "
             f"or teleport between one pair of adjacent tiles? (3) is the green square "
             f"present in every tile, or does it vanish/appear? (4) any sudden lighting or "
             f"scene change or shape distortion?\n{TAX_SCHEMA}")
        parsed, _ = V.ask_nova([{"text": p}, V.img_block(sheet)], max_tokens=900)
        time.sleep(0.3)
        out += (parsed or {}).get("findings", [])
    return out


def semantic_check(frames, fps, prompt_text):
    n = len(frames)
    idxs = [int(i * n / 8) for i in range(8)]
    sheet = V.contact_sheet([(frames[i], i / fps) for i in idxs], cols=4, tile_w=240)
    p = (f"Below are 8 frames uniformly sampled from an AI-generated video. The requested "
         f"generation prompt was:\n\"{prompt_text}\"\n"
         f"Rate how well the video content matches the prompt (subject, scene, colors, "
         f"motion), 1=completely different, 5=fully matches.\n"
         f'Respond ONLY with JSON: {{"alignment": 1-5, "mismatches": ["<item>", ...]}}')
    parsed, _ = V.ask_nova([{"text": p}, V.img_block(sheet)])
    return parsed or {"alignment": 1, "mismatches": ["unparseable"]}


def intercepted(findings, ev):
    c = (ev["start_s"] + ev["end_s"]) / 2
    for fd in findings:
        try:
            if fd["type"] in ALIAS[ev["type"]] and \
               float(fd["start_s"]) - TOL_S <= c <= float(fd["end_s"]) + TOL_S:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def run():
    gt_all = json.load(open(os.path.join(VIDEOS, "taxonomy_ground_truth.json")))
    records = []
    for g in gt_all:
        name = g["video"]
        frames, fps = V.read_frames(os.path.join(VIDEOS, name))
        d = json.load(open(os.path.join(PROBES, name.replace(".mp4", ".json"))))
        if "anomaly_score" not in d:
            raise SystemExit(f"run e2_fuse first for {name}")
        hard = detector_findings(d, fps) + deformation_findings(d, fps)
        subj_f = subject_vlm_tax(d, frames, fps)
        covered = [(f["start_s"] + f["end_s"]) / 2 for f in hard + subj_f]
        soft_f = soft_window_vlm_tax(d, frames, fps, covered)
        findings = hard + subj_f + soft_f
        hits = [intercepted(findings, ev) for ev in g["events"]]
        records.append({"video": name, "events": g["events"], "hits": hits,
                        "findings": findings})
        print(f"{name:18s} intercept={sum(hits)}/{len(hits)} "
              f"types={sorted(set(f['type'] for f in findings))}")

    # T10 语义不符（clean 视频 + 错 prompt）+ 对照（对 prompt）
    frames, fps = V.read_frames(os.path.join(VIDEOS, "tax_clean.mp4"))
    sem_cases = [
        ("semantic_A", "A blue cube bouncing on a white floor in an empty bright room.", True),
        ("semantic_B", "A red sports car driving on a highway at night, city lights in the "
                       "background.", True),
        ("semantic_control", GEN_PROMPT_TAX, False),
    ]
    for tag, ptxt, should_flag in sem_cases:
        r = semantic_check(frames, fps, ptxt)
        flag = int(r.get("alignment", 1)) <= 2
        ok = flag == should_flag
        records.append({"video": "tax_clean.mp4", "case": tag, "alignment": r,
                        "flagged": flag, "expected_flag": should_flag, "correct": ok})
        print(f"{tag:18s} alignment={r.get('alignment')} flagged={flag} "
              f"expected={should_flag} -> {'OK' if ok else 'WRONG'}")
        time.sleep(0.3)

    json.dump(records, open(os.path.join(RESULTS, "e6_interception.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
