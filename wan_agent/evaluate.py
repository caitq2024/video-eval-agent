#!/usr/bin/env python3
"""视频质量评估 Agent：对生成成片跑全帧探针 → detector 直判 → VLM 裁决 → T10 对齐。

流程（每阶段计时，前端要展示评估 agent 运行时长）：
  T0   文件层 gate（ffprobe）
  S1   fast_scan 全帧信号（GPU）
  S2   detector 直判：T6 黑帧 / T3 冻结(带 motion 先验) / T2 闪烁 / T1 跳帧 /
       T4 意外切换(分镜转场点豁免)
  S3   GroundingDINO 主体探针：T5 出界候选 / T7 变形 / T8 色相身份漂移(逐分镜参考)
  S4   软信号融合 → Top-K 候选窗 → Claude+rubric 定向裁决
  S5   T10 语义对齐：逐分镜 8 帧拼图 vs wan_prompt 评分
输出: wan_outputs/<pid>/<model>/eval.json（前端直接消费）

用法: /opt/pytorch/bin/python3 evaluate.py <pid> <model> [--device cuda:0] [--no-vlm]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVAL_ROOT, FFMPEG, OUT_ROOT, ask_claude, ffprobe_meta

SCRIPTS = os.path.join(EVAL_ROOT, 'experiments', 'scripts')
sys.path.insert(0, SCRIPTS)

# ---- 阈值：合成视频校准值 × 真实生成视频底噪修正（见 calibrate.py 输出与 REPORT_R2）----
TH = {
    'black_lum': 12.0,        # T6 亮度
    'black_run': 2,
    'freeze_d1': 0.35,        # T3 帧差（Wan 有纹理噪声，0.05 合成值太严；clean 校准后上调）
    'freeze_run_s': 1.0,      # 持续 1s 才算冻结
    'flicker': 6.0,           # T2 孤立亮度尖峰（clean max ×1.5 校准）
    'warp': 6.0,              # T1 warp 残差（clean max ×1.5 校准）
    'cut_hsv': 27.0,          # T4 HSV 帧差（ContentDetector 同款）
    'trans_margin_s': 0.5,    # 转场豁免边界
    'deform_logar': 0.35,     # T7 bbox 纵横比 log 偏移
    'hue_drift': 0.5,         # T8 色相直方图余弦距离
    'miss_run': 3,            # T5 连续缺失子采样帧数
    'det_th': 0.30,           # GroundingDINO 置信度
    'fuse_top_k': 3,
}
SUB_EVERY = 3                 # 主体探针子采样步长


# ---------------------------------------------------------------- helpers
def shot_spans(meta, model, film_dur):
    """成片时间轴上每个分镜的 [start, end]（以转场中点为界）。"""
    trans = meta.get('films', {}).get(model, {}).get('transitions', [])
    bounds = [0.0] + [(t['start_s'] + t['end_s']) / 2 for t in trans] + [film_dur]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def exempt_mask(n, fps, trans, margin):
    """转场豁免帧掩码（T4/T1/T2 在这些帧不直判）。"""
    m = np.zeros(n, bool)
    for t in trans:
        lo = max(0, int((t['start_s'] - margin) * fps))
        hi = min(n, int((t['end_s'] + margin) * fps) + 1)
        m[lo:hi] = True
    return m


def runs_where(cond):
    """布尔数组 → [(start_idx, end_idx)] 连续段（闭开区间）。"""
    out, i, n = [], 0, len(cond)
    while i < n:
        if cond[i]:
            j = i
            while j < n and cond[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def align_n(x, n):
    """信号对齐到 n 帧：diff_d1 是 n-1（前补 0），clip/warp 是子采样长度（重复展开）。"""
    a = np.asarray(x, np.float32)
    if len(a) == n:
        return a
    if len(a) == n - 1:
        return np.concatenate([[0], a])
    r = int(np.ceil(n / max(1, len(a))))
    out = np.repeat(a, r)
    return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)), 'edge')


def shot_of(t, spans):
    for k, (a, b) in enumerate(spans):
        if a <= t < b:
            return k
    return len(spans) - 1


def downsample_preview(signals, cuts, n, fps, max_pts=480):
    """信号降采样（max-pool 保尖峰），前端画时间轴用。"""
    stride = max(1, int(np.ceil(n / max_pts)))
    prev = {}
    for k, v in signals.items():
        a = align_n(v, n)
        pad = (-len(a)) % stride
        if pad:
            a = np.concatenate([a, np.full(pad, a[-1] if len(a) else 0)])
        prev[k] = np.round(a.reshape(-1, stride).max(1), 3).tolist()
    return {'stride': stride, 'fps': fps, 'series': prev,
            'cuts_s': [round(c / fps, 2) for c in cuts]}


# ---------------------------------------------------------------- S2 detectors
def detect_hard(sig, cuts, fps, meta, model, spans, trans):
    n = len(sig['luminance'])
    lum = np.asarray(sig['luminance'], np.float32)
    d1 = align_n(sig['diff_d1'], n)
    d1[0] = d1[1] if n > 1 else 1.0          # 补位帧不参与冻结判定
    flick = np.asarray(sig['flicker'], np.float32)
    warp = np.asarray(sig['warp_residual'], np.float32)
    ex = exempt_mask(n, fps, trans, TH['trans_margin_s'])
    shots = meta['shots']
    F = []

    # T6 黑帧
    for a, b in runs_where(lum < TH['black_lum']):
        if b - a >= TH['black_run']:
            F.append(dict(type='T6_black', start_s=a / fps, end_s=b / fps, severity=5,
                          evidence=f'亮度<{TH["black_lum"]} 持续 {b - a} 帧（全黑）',
                          confidence=0.99, verdict_by='detector'))
    # T3 冻结（低运动分镜先验：static+low 需 2 倍时长）
    for a, b in runs_where(d1 < TH['freeze_d1']):
        need = TH['freeze_run_s'] * fps
        s = shots[min(shot_of(a / fps, spans), len(shots) - 1)]
        if s.get('camera') == 'static' and s.get('motion_level') == 'low':
            need *= 2
        if b - a >= need and not ex[a:b].all():
            F.append(dict(type='T3_freeze', start_s=a / fps, end_s=b / fps, severity=4,
                          evidence=f'帧差<{TH["freeze_d1"]} 持续 {round((b - a) / fps, 2)}s，'
                                   f'画面完全静止（该分镜 camera={s.get("camera")}/'
                                   f'motion={s.get("motion_level")}，已用 2 倍时长门槛仍触发）',
                          confidence=0.9, verdict_by='detector'))
    # T2 闪烁（孤立尖峰，避开转场/切点邻域）
    cutset = set(cuts)
    for i in np.where(flick > TH['flicker'])[0]:
        if ex[i] or any(abs(int(i) - c) <= 2 for c in cutset):
            continue
        F.append(dict(type='T2_flicker', start_s=i / fps, end_s=(i + 1) / fps, severity=3,
                      evidence=f'孤立亮度尖峰 {flick[i]:.1f}>{TH["flicker"]}（单帧闪烁）',
                      confidence=0.85, verdict_by='detector'))
    # T1 跳帧：真实生成视频运动大、warp 底噪高（wan2.1 中位数~5），
    # 绝对阈值会把整段高运动误判 —— 改判「孤立尖峰」：相对局部中位数 3 倍以上且邻帧回落
    if len(warp) >= 5:
        k_med = 7
        pad_w = np.pad(warp, k_med, 'edge')
        local_med = np.array([np.median(pad_w[i:i + 2 * k_med + 1])
                              for i in range(len(warp))])
        for k in np.where(warp > np.maximum(TH['warp'], 3.0 * local_med))[0]:
            k = int(k)
            left = warp[k - 1] if k > 0 else 0
            right = warp[k + 1] if k + 1 < len(warp) else 0
            if max(left, right) > 0.6 * warp[k]:
                continue                     # 持续高 warp = 运动，不是跳帧
            i = k * 3                        # fast_scan SUB_EVERY=3
            if i >= n or ex[min(i, n - 1)] or lum[min(i, n - 1)] < TH['black_lum']:
                continue
            if any(abs(i - c) <= 4 for c in cutset):
                continue
            F.append(dict(type='T1_jump', start_s=i / fps, end_s=min(n, i + 3) / fps,
                          severity=4,
                          evidence=f'光流 warp 残差 {warp[k]:.1f} 孤立尖峰'
                                   f'（局部中位数的 {warp[k] / (local_med[k] + 1e-6):.1f} 倍，'
                                   f'邻帧回落，内容瞬移）',
                          confidence=0.85, verdict_by='detector'))
    # T4 意外切换（转场豁免）
    for c in cuts:
        if not ex[min(c, n - 1)]:
            F.append(dict(type='T4_unexpected_cut', start_s=c / fps, end_s=(c + 1) / fps,
                          severity=4,
                          evidence='HSV 帧差>27 的硬切，且不在分镜计划的转场点上',
                          confidence=0.9, verdict_by='detector'))
    for f in F:
        f['start_s'] = round(f['start_s'], 2)
        f['end_s'] = round(f['end_s'], 2)
    return merge_findings(F)


def merge_findings(F, gap_s=1.0):
    """同类型且时间相邻的 finding 合并为一个事件窗。"""
    out = []
    for f in sorted(F, key=lambda x: (x['type'], x['start_s'])):
        if out and out[-1]['type'] == f['type'] and \
                f['start_s'] - out[-1]['end_s'] <= gap_s:
            out[-1]['end_s'] = max(out[-1]['end_s'], f['end_s'])
            out[-1]['severity'] = max(out[-1]['severity'], f['severity'])
        else:
            out.append(dict(f))
    return out


# ---------------------------------------------------------------- S3 subject probes
def decode_sub(path, w=640, h=360):
    import subprocess
    cmd = [FFMPEG, '-loglevel', 'error', '-i', path,
           '-vf', f'select=not(mod(n\\,{SUB_EVERY})),scale={w}:{h}', '-vsync', '0',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)


_GD_CACHE = {}          # device -> (processor, model)，批量评估跨视频复用


def subject_probes(film, meta, model, spans, fps, device):
    """GroundingDINO 检测 expected_subjects → T5 候选/T7 变形/T8 色相漂移。"""
    import cv2
    import torch
    if device not in _GD_CACHE:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        _GD_CACHE[device] = (
            AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'),
            GroundingDinoForObjectDetection.from_pretrained(
                'IDEA-Research/grounding-dino-tiny').to(device).eval())
    proc, gd = _GD_CACHE[device]

    frames = decode_sub(film)
    n = len(frames)
    sub_fps = fps / SUB_EVERY
    subjects = []
    for s in meta['shots']:
        for x in s.get('expected_subjects', []):
            if x.lower() not in [y.lower() for y in subjects]:
                subjects.append(x)
    if not subjects:
        return [], {}
    text = '. '.join(s.rstrip('.').lower() for s in subjects) + '.'

    dets = {s: [None] * n for s in subjects}
    B = 16
    with torch.no_grad():
        for i in range(0, n, B):
            batch = [frames[j] for j in range(i, min(i + B, n))]
            inp = proc(images=batch, text=[text] * len(batch), return_tensors='pt').to(device)
            with torch.autocast('cuda'):
                out = gd(**inp)
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=TH['det_th'], text_threshold=0.25,
                target_sizes=[(360, 640)] * len(batch))
            for k, r in enumerate(res):
                for lb, sc, bx in zip(r['labels'], r['scores'], r['boxes']):
                    for s in subjects:
                        key_words = [w for w in s.lower().split() if len(w) > 2]
                        if any(w in lb for w in key_words):
                            if dets[s][i + k] is None or sc > dets[s][i + k][0]:
                                dets[s][i + k] = (float(sc), [float(v) for v in bx])
    torch.cuda.empty_cache()

    findings, cand = [], []
    track_rates = {}
    for subj in subjects:
        # 该主体应出现的分镜集合
        want = [k for k, s in enumerate(meta['shots'])
                if any(subj.lower() == x.lower() for x in s.get('expected_subjects', []))]
        boxes = dets[subj]
        present = np.array([b is not None for b in boxes])
        for k in want:
            a, b = spans[min(k, len(spans) - 1)]
            lo, hi = int(a * sub_fps), min(n, int(b * sub_fps))
            if hi - lo < 4:
                continue
            seg = present[lo:hi]
            rate = float(seg.mean())
            track_rates[f'{subj}@shot{k + 1}'] = round(rate, 2)
            # T5 候选：先出现，后连续缺失
            if seg[:max(3, len(seg) // 4)].any():
                miss = runs_where(~seg)
                for ma, mb in miss:
                    if mb - ma >= TH['miss_run'] and (ma > 0 or rate > 0.5):
                        cand.append(dict(type='T5_out_of_frame_candidate', subject=subj,
                                         start_s=round((lo + ma) / sub_fps, 2),
                                         end_s=round((lo + mb) / sub_fps, 2), shot=k + 1))
            elif rate < 0.2:
                # 整个分镜基本没检出 → 语义缺主体，交 T10/VLM
                cand.append(dict(type='T5_subject_absent', subject=subj,
                                 start_s=round(a, 2), end_s=round(b, 2), shot=k + 1))
            # T7 变形 & T8 色相漂移（分镜内参考）
            idxs = [i for i in range(lo, hi) if boxes[i] is not None]
            if len(idxs) < 6:
                continue
            ar = []
            hists = []
            for i in idxs:
                x0, y0, x1, y1 = boxes[i][1]
                w, h = max(1, x1 - x0), max(1, y1 - y0)
                border = x0 < 5 or y0 < 5 or x1 > 635 or y1 > 355
                ar.append((np.log(w / h), border))
                c = frames[i][max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
                hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV) if c.size >= 300 else None
                # crop 质量门：太小或低饱和（暗色装备/灯光）的色相直方图不可靠
                if hsv is None or c.shape[0] * c.shape[1] < 1500 or \
                        hsv[..., 1].mean() < 25:
                    hists.append(None)
                    continue
                hh = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                                  [0, 180, 0, 256]).flatten()
                hists.append(hh / (hh.sum() + 1e-8))
            # 活体主体姿态变化会自然改变 bbox 纵横比 —— 按 motion_level 放宽阈值，
            # 且要求连续 3 个子采样帧（~0.5s）持续偏移才算病理性变形
            motion = meta['shots'][k].get('motion_level', 'medium')
            de_th = TH['deform_logar'] * {'low': 1.0, 'medium': 1.6, 'high': 2.2}[motion]
            full = [v for v, bd in ar if not bd]
            if len(full) >= 6:
                med = np.median(full[:max(3, len(full) // 3)])
                run = 0
                for j, (v, bd) in enumerate(ar):
                    if not bd and abs(v - med) > de_th:
                        run += 1
                        if run == 3:
                            # 姿态/透视也会改变纵横比，像素证据不充分 → VLM 确认
                            cand.append(dict(
                                type='T7_deform_candidate', subject=subj, shot=k + 1,
                                start_s=round(idxs[j - 2] / sub_fps, 2),
                                end_s=round((idxs[j] + 1) / sub_fps, 2),
                                signal=f'bbox 纵横比 log 偏移 {abs(v - med):.2f}'
                                       f'>{de_th:.2f} 且持续（motion={motion}）'))
                            break
                    else:
                        run = 0
            hv = [(i, hh) for i, hh in zip(idxs, hists) if hh is not None]
            if len(hv) >= 6:
                ref_n = max(2, min(len(hv) // 3, int(sub_fps)))
                href = np.median(np.stack([h for _, h in hv[:ref_n]]), 0)
                href /= (np.linalg.norm(href) + 1e-8)
                run = 0
                for i, hh in hv[ref_n:]:
                    dcos = 1 - float(hh @ href) / (np.linalg.norm(hh) + 1e-8)
                    if dcos > TH['hue_drift']:
                        run += 1
                        if run == 3:
                            # 光照/入烟/滤镜变化也会漂移色相 → VLM 确认是否真变身份
                            cand.append(dict(
                                type='T8_drift_candidate', subject=subj, shot=k + 1,
                                start_s=round((i - 2) / sub_fps, 2),
                                end_s=round((i + 1) / sub_fps, 2),
                                signal=f'色相直方图余弦距离 {dcos:.2f}'
                                       f'>{TH["hue_drift"]} 且持续'))
                            break
                    else:
                        run = 0
    # 逐子采样帧的主体在场轨迹（供 VLM 裁决交叉验证：像素证据优先于 VLM 印象）
    presence = {s: [b is not None for b in dets[s]] for s in subjects}
    return findings, {'candidates': cand, 'track_rates': track_rates,
                      'presence': presence, 'sub_fps': sub_fps}


# ---------------------------------------------------------------- S4/S5 VLM
def make_rubric(meta):
    shots_txt = '\n'.join(f'Shot {s["shot_id"]}: {s["wan_prompt"]}' for s in meta['shots'])
    p = ('You are designing evaluation criteria for an AI-generated multi-shot video '
         'BEFORE seeing it (to avoid being biased by the output). The storyboard:\n'
         f'{shots_txt}\n\nWrite 4-6 hard pass/fail criteria focused on: subject identity '
         'consistency within each shot, subject visibility, temporal continuity inside '
         'a shot (no unexpected cuts/jumps), and prompt adherence. '
         '每条标准用简体中文书写。'
         'Respond ONLY with JSON: {"criteria": ["<中文标准>", ...]}')
    parsed, _ = ask_claude([{'text': p}])
    return (parsed or {}).get('criteria', [])


def sheet_block(frames_ts):
    import vlm_common as V
    return V.img_block(V.contact_sheet(frames_ts, cols=4, tile_w=300))


def read_film_frames(path):
    import vlm_common as V
    return V.read_frames(path)


def vlm_window_verdict(frames, fps, win, meta, rubric, context):
    n = len(frames)
    c = int((win['start_s'] + win['end_s']) / 2 * fps)
    lo = max(0, min(n - 8, c - 4))
    step = max(1, int((win['end_s'] - win['start_s']) * fps / 8)) or 1
    ids = [min(n - 1, lo + i * step) for i in range(8)]
    rub = '\n'.join(f'- {x}' for x in rubric)
    p = (f'You are inspecting an AI-generated video (storyboard-based, planned transitions '
         f'are OK). Intended idea: "{meta["idea"]}"\n'
         f'Pre-registered criteria:\n{rub}\n\n'
         f'A detector flagged t={win["start_s"]}-{win["end_s"]}s: {context}. '
         f'Below are 8 frames from that region (timestamps burned in).\n'
         f'Judge SPECIFICALLY: is there a real quality defect there (subject discontinuity, '
         f'unintended exit, morphing, artifact), or is it acceptable/intentional?\n'
         f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
         f'"type": "<short defect type or none>", "severity": 1-5, '
         f'"reason": "<简体中文简述，给中国客户看>", "confidence": 0.0-1.0}}')
    frames_ts = [(frames[k], k / fps) for k in ids]
    parsed, raw = ask_claude([{'text': p}, sheet_block(frames_ts)])
    return parsed or {'verdict': 'PARSE_FAIL', 'reason': raw[:120]}


def vlm_t10_alignment(frames, fps, span, shot, rubric):
    a, b = span
    n = len(frames)
    ids = [min(n - 1, int((a + (b - a) * (i + 0.5) / 8) * fps)) for i in range(8)]
    rub = '\n'.join(f'- {x}' for x in rubric)
    p = (f'Rate how well this AI-generated shot matches its intended prompt.\n'
         f'Intended prompt: "{shot["wan_prompt"]}"\n'
         f'Pre-registered criteria:\n{rub}\n\n'
         f'Below are 8 frames sampled uniformly from the shot.\n'
         f'Score 1-5: 5=faithful, 3=partial (subject/action/style deviations), '
         f'1=unrelated or subject missing. Also list missing elements.\n'
         f'missing 与 reason 用简体中文书写（给中国客户看）。\n'
         f'Respond ONLY with JSON: {{"score": 1-5, "missing": ["<中文>"], '
         f'"reason": "<中文简述>"}}')
    frames_ts = [(frames[k], k / fps) for k in ids]
    parsed, raw = ask_claude([{'text': p}, sheet_block(frames_ts)])
    return parsed or {'score': None, 'reason': raw[:120]}


# ---------------------------------------------------------------- scoring
def score_findings(findings, t10):
    per_dim = {'temporal': 100.0, 'structural': 100.0, 'subject': 100.0, 'semantic': 100.0}
    dim_of = {'T1_jump': 'temporal', 'T2_flicker': 'temporal', 'T3_freeze': 'temporal',
              'T4_unexpected_cut': 'structural', 'T6_black': 'structural',
              'T5_out_of_frame': 'subject', 'T7_deform': 'subject',
              'T8_identity_drift': 'subject', 'T9_vanish': 'subject',
              'T10_misalignment': 'semantic', 'vlm_defect': 'subject'}
    hard = []
    for f in findings:
        d = dim_of.get(f['type'], 'subject')
        # 惩罚随缺陷持续时长放大：整镜头冻结(4s+)要比 0.2s 尖峰重得多
        dur = max(0.0, f.get('end_s', 0) - f.get('start_s', 0))
        dur_factor = min(3.0, max(1.0, dur / 1.5))
        pen = f.get('severity', 3) * 6 * dur_factor
        per_dim[d] = max(0.0, per_dim[d] - pen)
        if f.get('severity', 3) >= 4:
            hard.append(f['type'])
    scores = [t.get('score') for t in t10 if t.get('score')]
    if scores:
        per_dim['semantic'] = min(per_dim['semantic'], (np.mean(scores) / 5) * 100)
        if min(scores) <= 2:
            hard.append('T10_misalignment')
    # 总分 = 0.5×均值 + 0.5×最差维度：单一维度重伤不能被其余三维平均回「合格」
    vals = list(per_dim.values())
    total = round(0.5 * float(np.mean(vals)) + 0.5 * float(np.min(vals)), 1)
    return {'per_dim': {k: round(v, 1) for k, v in per_dim.items()},
            'hard_fails': sorted(set(hard)), 'total': total}


# ---------------------------------------------------------------- main
def evaluate(pid, model, device='cuda:0', use_vlm=True):
    pdir = os.path.join(OUT_ROOT, pid)
    meta = json.load(open(os.path.join(pdir, 'shots.json')))
    film = os.path.join(pdir, model, 'film.mp4')
    timing, vlm_calls = {}, 0
    t_all = time.time()

    # T0
    t0 = time.time()
    gate = ffprobe_meta(film)
    clips_gate = {}
    for s in meta['shots']:
        cp = os.path.join(pdir, model, f'shot{s["shot_id"]}.mp4')
        clips_gate[f'shot{s["shot_id"]}'] = ffprobe_meta(cp) if os.path.exists(cp) \
            else {'decodable': False, 'missing': True}
    timing['t0_gate_s'] = round(time.time() - t0, 2)
    if not gate['decodable']:
        return {'video': film, 'model': model, 'prompt_id': pid,
                'findings': [{'type': 'T0_file_gate', 'severity': 5,
                              'evidence': 'film not decodable', 'verdict_by': 'detector'}],
                'scores': {'per_dim': {}, 'hard_fails': ['T0_file_gate'], 'total': 0},
                'timing': timing}

    # S1 fast_scan
    t0 = time.time()
    from fast_scan import scan
    sc = scan(film, device)
    fps, n = sc['fps'], sc['n_frames']
    timing['s1_scan_s'] = round(time.time() - t0, 2)
    timing['s1_scan_compute_s'] = sc['compute_s']

    trans = meta.get('films', {}).get(model, {}).get('transitions', [])
    spans = shot_spans(meta, model, sc['duration_s'])

    # S2 direct detectors
    t0 = time.time()
    findings = detect_hard(sc['signals'], sc['cuts_frames'], fps, meta, model, spans, trans)
    timing['s2_detect_s'] = round(time.time() - t0, 2)

    # S3 subject probes
    t0 = time.time()
    try:
        sub_findings, sub_info = subject_probes(film, meta, model, spans, fps, device)
    except Exception as e:
        sub_findings, sub_info = [], {'error': str(e)[:300]}
    findings = merge_findings(findings + sub_findings)
    timing['s3_probes_s'] = round(time.time() - t0, 2)

    # S4 fusion windows + VLM adjudication
    t0 = time.time()
    from e2_fuse import candidate_windows, zscore
    sig = sc['signals']

    def to_n(x):
        return align_n(x, n)

    soft = np.zeros(n, np.float32)
    for k, w, floor in [('flicker', 1.0, TH['flicker'] * 0.6),
                        ('diff_d1', 0.5, 12.0), ('clip_dist', 1.0, 0.15)]:
        raw = to_n(sig[k])
        soft += w * np.clip(zscore(raw), 0, 8) * (raw > floor)
    warp_full = to_n(sig['warp_residual'])
    soft += 1.5 * np.clip(zscore(warp_full), 0, 8) * (warp_full > TH['warp'] * 0.7)
    ex = exempt_mask(n, fps, trans, TH['trans_margin_s'])
    soft[ex] = 0
    wins = candidate_windows(soft, fps, top_k=TH['fuse_top_k'], thresh=3.0)
    timing['s4_fuse_s'] = round(time.time() - t0, 2)

    rubric = []
    t10 = []
    vlm_rejected = []
    if use_vlm:
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()
        rubric = make_rubric(meta)               # rubric 先行，其余裁决都要引用它
        vlm_calls += 1
        frames, ffps = read_film_frames(film)

        # 主体探针候选映射（T5 缺失 / T7 变形 / T8 漂移 —— 像素证据不充分的都走这里）
        CAND_MAP = {
            'T5_out_of_frame_candidate': ('T5_out_of_frame',
                'subject "{subject}" not detected in this window of shot {shot} '
                '(expected visible per storyboard) — unintended exit/crop, or '
                'intentional framing?'),
            'T5_subject_absent': ('T5_out_of_frame',
                'subject "{subject}" was almost never detected in shot {shot} '
                '(expected per storyboard) — is it truly absent?'),
            'T7_deform_candidate': ('T7_deform',
                'subject "{subject}" bounding-box aspect changed a lot in shot {shot} '
                '({signal}) — pathological deformation/morphing, or normal '
                'pose/perspective change?'),
            'T8_drift_candidate': ('T8_identity_drift',
                'subject "{subject}" color histogram drifted in shot {shot} '
                '({signal}) — identity/appearance change (e.g. object morphs, '
                'clothing/color swap), or normal lighting change?'),
        }

        presence = sub_info.get('presence', {})
        p_sub_fps = sub_info.get('sub_fps')

        def window_presence(w):
            """窗口内各主体的跟踪检出率（像素证据，用于约束/否决 VLM 印象）。"""
            if not presence or not p_sub_fps:
                return {}
            lo = max(0, int(w['start_s'] * p_sub_fps))
            hi = max(lo + 1, int(w['end_s'] * p_sub_fps))
            return {s: sum(arr[lo:hi]) / max(1, len(arr[lo:hi]))
                    for s, arr in presence.items() if arr[lo:hi]}

        def judge_window(w):
            wp = window_presence(w)
            ctx = f'fused anomaly score {w["score"]} (soft signals).'
            if wp:
                ctx += (' Object-tracker detection rate in this window: '
                        + ', '.join(f'"{s}" {r * 100:.0f}%' for s, r in wp.items())
                        + '. A subject at ~100% IS present in every frame — do NOT '
                          'claim it disappeared or went missing.')
            v = vlm_window_verdict(frames, ffps, w, meta, rubric, ctx)
            if v.get('verdict') == 'defect':
                # 交叉否决：VLM 声称主体消失，但单实例主体跟踪检出率满格 → 幻觉，拒绝
                claim = (str(v.get('type', '')) + str(v.get('reason', ''))).lower()
                vanish_kw = ['消失', '不见', 'disappear', 'vanish', 'missing',
                             'absent', 'discontinu']
                singular = {s: r for s, r in wp.items()
                            if s.lower().startswith(('a ', 'an '))}
                if any(k in claim for k in vanish_kw) and singular and \
                        min(singular.values()) >= 0.95:
                    vlm_rejected.append(dict(
                        window=[w['start_s'], w['end_s']],
                        vlm_claim=str(v.get('reason', ''))[:150],
                        rejected_because='跟踪器证据否决：窗口内单实例主体检出率 '
                        + ', '.join(f'{s}={r * 100:.0f}%' for s, r in singular.items())))
                    return None
                return dict(type='vlm_defect', start_s=w['start_s'], end_s=w['end_s'],
                            severity=int(v.get('severity', 3)),
                            evidence=f'{v.get("type")}: {v.get("reason", "")}'[:200],
                            confidence=float(v.get('confidence', 0.6)), verdict_by='vlm')

        def judge_candidate(c):
            ftype, tpl = CAND_MAP[c['type']]
            win = {'start_s': c['start_s'], 'end_s': max(c['end_s'], c['start_s'] + 0.5)}
            ctx = tpl.format(**{k: c.get(k, '') for k in ('subject', 'shot', 'signal')})
            v = vlm_window_verdict(frames, ffps, win, meta, rubric, ctx)
            if v.get('verdict') == 'defect':
                return dict(type=ftype, start_s=c['start_s'], end_s=c['end_s'],
                            severity=int(v.get('severity', 3)),
                            evidence=f'{c["subject"]}（分镜{c["shot"]}，'
                                     f'{c.get("signal", "检测缺失")}）：'
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.6)), verdict_by='vlm')

        def judge_t10(k_s):
            k, s = k_s
            r = vlm_t10_alignment(frames, ffps, spans[k], s, rubric)
            return {'shot_id': s['shot_id'],
                    **{kk: r.get(kk) for kk in ('score', 'missing', 'reason')}}

        T10_VOTES = 3        # Bedrock 图像输入非严格确定，边界 2/3 分会翻转 → 三票取中位

        # 除 rubric 外全部 VLM 调用并行（Bedrock 侧无依赖）
        with ThreadPoolExecutor(max_workers=16) as pool:
            fut_wins = [pool.submit(judge_window, w) for w in wins]
            fut_cands = [pool.submit(judge_candidate, c)
                         for c in sub_info.get('candidates', [])]
            fut_t10 = [(k, [pool.submit(judge_t10, (k, s)) for _ in range(T10_VOTES)])
                       for k, s in enumerate(meta['shots']) if k < len(spans)]
            for f in fut_wins + fut_cands:
                vlm_calls += 1
                r = f.result()
                if r:
                    findings.append(r)
            for k, futs in fut_t10:
                votes = [f.result() for f in futs]
                vlm_calls += len(votes)
                scored = sorted([v for v in votes if v.get('score')],
                                key=lambda v: v['score'])
                if scored:
                    med = scored[len(scored) // 2]
                    med['votes'] = [v.get('score') for v in votes]
                    t10.append(med)
                else:
                    t10.append(votes[0])
        t10.sort(key=lambda x: x['shot_id'])
        timing['s4_vlm_s'] = round(time.time() - t0, 2)

    timing['total_s'] = round(time.time() - t_all, 2)
    timing['vlm_calls'] = vlm_calls

    out = {
        'video': os.path.relpath(film, OUT_ROOT), 'model': model, 'prompt_id': pid,
        'idea': meta['idea'], 'title': meta.get('title'),
        'shots': meta['shots'], 'transitions': trans,
        'shot_spans_s': [[round(a, 2), round(b, 2)] for a, b in spans],
        'file_gate': {'film': gate, 'clips': clips_gate},
        'findings': findings, 't10_alignment': t10,
        'subject_probe': sub_info.get('track_rates', {}),
        'probe_candidates': sub_info.get('candidates', []),
        'vlm_rejected_by_tracker': vlm_rejected,
        'rubric': rubric,
        'signals_preview': downsample_preview(sc['signals'], sc['cuts_frames'], n, fps),
        'scan_meta': {k: sc[k] for k in ('n_frames', 'fps', 'duration_s',
                                         'compute_s', 'realtime_factor')},
        'scores': score_findings(findings, t10),
        'timing': timing,
    }
    ep = os.path.join(pdir, model, 'eval.json')
    json.dump(out, open(ep, 'w'), ensure_ascii=False, indent=1)
    print(f'[eval] {pid}/{model} total={out["scores"]["total"]} '
          f'findings={len(findings)} hard={out["scores"]["hard_fails"]} '
          f'time={timing["total_s"]}s (vlm {vlm_calls} calls)')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pid')
    ap.add_argument('model')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-vlm', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('HF_HOME', '/home/ec2-user/hf_cache')
    os.environ.setdefault('TORCH_HOME', '/home/ec2-user/torch_cache')
    evaluate(a.pid, a.model, a.device, not a.no_vlm)
