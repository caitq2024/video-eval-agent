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
    shot_hue = {}                # subj -> {shot: 中位色相直方图}（T19 跨镜头比对用）
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
            if len(hv) >= 4:
                shot_hue.setdefault(subj, {})[k + 1] = np.median(
                    np.stack([h for _, h in hv]), 0)
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
            # T11 信号 C：主体轨迹动力学 —— 相邻子采样帧 bbox IoU 骤降 + 中心瞬移
            # （区别于 T8 的对参考慢漂移；直接回答"哪个主体在哪一帧突变"）
            # 只对单实例主体启用：复数主体（"fencing masks"）检测框会在实例间跳，
            # IoU=0 是跟踪歧义而非瞬移
            if not subj.lower().startswith(('a ', 'an ')):
                continue
            jumps = []
            prev_i = None
            for i in idxs:
                if prev_i is not None and i - prev_i <= 2:
                    b0, b1 = boxes[prev_i][1], boxes[i][1]
                    ix0, iy0 = max(b0[0], b1[0]), max(b0[1], b1[1])
                    ix1, iy1 = min(b0[2], b1[2]), min(b0[3], b1[3])
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    a0 = (b0[2] - b0[0]) * (b0[3] - b0[1])
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    iou = inter / (a0 + a1 - inter + 1e-6)
                    diag = max(np.hypot(b1[2] - b1[0], b1[3] - b1[1]), 1.0)
                    dist = np.hypot((b1[0] + b1[2] - b0[0] - b0[2]) / 2,
                                    (b1[1] + b1[3] - b0[1] - b0[3]) / 2)
                    jumps.append((prev_i, i, iou, dist / diag))
                prev_i = i
            if len(jumps) >= 6:
                ratios = np.array([r for *_, r in jumps])
                medr = float(np.median(ratios)) + 1e-6
                mot_mult = {'low': 1.0, 'medium': 1.5, 'high': 2.2}[motion]
                for pi, i, iou, r in jumps:
                    if iou < 0.1 and r > max(0.8 * mot_mult, 4 * medr):
                        u = [min(boxes[pi][1][0], boxes[i][1][0]),
                             min(boxes[pi][1][1], boxes[i][1][1]),
                             max(boxes[pi][1][2], boxes[i][1][2]),
                             max(boxes[pi][1][3], boxes[i][1][3])]
                        cand.append(dict(
                            type='T11_local_candidate', subject=subj, shot=k + 1,
                            start_s=round(pi / sub_fps, 2), end_s=round(i / sub_fps, 2),
                            film_frames=[pi * SUB_EVERY, i * SUB_EVERY],
                            region_640x360=[round(v, 1) for v in u],
                            signal=f'相邻帧 bbox IoU={iou:.2f}<0.1 且中心瞬移 '
                                   f'{r:.2f} 倍对角线（本镜中位 {medr:.2f}，'
                                   f'motion={motion}）'))
                        break        # 每主体每分镜最多报一处，交 VLM 精查
    # 逐子采样帧的主体在场轨迹（供 VLM 裁决交叉验证：像素证据优先于 VLM 印象）
    presence = {s: [b is not None for b in dets[s]] for s in subjects}
    boxes = {s: [b[1] if b else None for b in dets[s]] for s in subjects}
    return findings, {'candidates': cand, 'track_rates': track_rates,
                      'presence': presence, 'sub_fps': sub_fps,
                      'shot_hue': shot_hue, 'boxes': boxes}


def detect_t11_blocks(sc, fps, ex, cuts, spans, shots, top_k=3):
    """T11 信号 A：warp 残差 8×8 块矩阵 → **逐块时间维** robust z。
    镜头运动会让边缘块残差恒高（clean 实测中位 38），跨块比较无效——每块只和
    自己的历史比；再要求全局 z 低（全局也高的走 T1）+ 孤立尖峰 + 豁免/运动先验。"""
    from e2_fuse import zscore
    wb = np.asarray(sc.get('warp_blocks', []), np.float32)      # T,64
    gm = np.asarray(sc['signals']['warp_residual'], np.float32)
    if wb.ndim != 2 or len(wb) < 10:
        return []
    med = np.median(wb, axis=0)
    mad = np.median(np.abs(wb - med), axis=0) + 0.5             # 0.5: 静块 MAD 下限
    z = (wb - med) / (1.4826 * mad)                             # T,64
    zb = z.max(axis=1)
    bidx = z.argmax(axis=1)
    zg = zscore(gm)
    n_film = len(sc['signals']['luminance'])
    active = wb > (med + 8.0)                                   # T,64 显著抬升掩码

    def has_trail(k, b):
        """运动物体跨块留轨迹：t±1 有「t 时不活跃」的新鲜邻块活跃 → 小目标在移动
        （RAFT 对小目标光流常失败，块尖峰与 glitch 相同，靠轨迹区分）。
        glitch 的足迹块在 t 和 t±1 是同一批，没有新鲜邻块。"""
        row, col = b // 8, b % 8
        nb = [r * 8 + c for r in (row - 1, row, row + 1)
              for c in (col - 1, col, col + 1)
              if 0 <= r < 8 and 0 <= c < 8 and r * 8 + c != b]
        for kk in (k - 1, k + 1):
            if 0 <= kk < len(wb):
                fresh = active[kk, nb] & ~active[k, nb]
                if fresh.any():
                    return True
        return False

    def run_len(k, b):
        """该块围绕 k 的连续高位长度（单帧 glitch 落在采样点会点亮相邻两个 warp 对）。"""
        hi = z[:, b] > 0.5 * zb[k]
        lo = k
        while lo > 0 and hi[lo - 1]:
            lo -= 1
        hh = k
        while hh + 1 < len(hi) and hi[hh + 1]:
            hh += 1
        return hh - lo + 1

    out = []
    taken = []                               # 已取候选的时间索引（去重）
    for idx in np.argsort(-z, axis=None):
        k, b = divmod(int(idx), 64)          # 逐 (时间,块) 格点选：同帧可能同时有
        if z[k, b] < 6.0 or len(out) >= top_k:   # 运动小目标(被否)和真 glitch
            break
        if k < 2:                            # warp[0] 是填充零行，邻域不完整
            continue
        if any(abs(k - t) <= 2 for t in taken):
            continue
        if zg[k] > 2.5:                      # 全局也异常 → T1 通道处理
            continue
        if wb[k, b] < med[b] + 8.0:          # 绝对下限：残差抬升要有物理量
            continue
        if run_len(k, b) > 2:                # 该块持续高（≥3 采样）= 局部运动而非突变
            continue
        if has_trail(k, b):                  # 新鲜邻块轨迹 = 小目标运动，非 glitch
            continue
        i = k * 3                            # fast_scan SUB_EVERY=3 → 帧号
        if i >= n_film or ex[min(i, n_film - 1)]:
            continue
        if any(abs(i - c) <= 4 for c in cuts):
            continue
        shot_k = shot_of(i / fps, spans)
        mot = shots[min(shot_k, len(shots) - 1)].get('motion_level', 'medium')
        if mot == 'high' and z[k, b] < 9.0:  # 高运动分镜提高门槛
            continue
        taken.append(k)
        xy = [round((b % 8 + 0.5) / 8, 3), round((b // 8 + 0.5) / 8, 3)]
        out.append(dict(
            type='T11_local_candidate', subject=None, shot=shot_k + 1,
            start_s=round(max(0, i - 3) / fps, 2), end_s=round(i / fps, 2),
            film_frames=[max(0, i - 3), i],
            region_xy_norm=xy,
            signal=f'warp 残差块时间维 z={z[k, b]:.1f}（该块中位 {med[b]:.0f}→'
                   f'{wb[k, b]:.0f}，全局 z={zg[k]:.1f} 低），孤立尖峰，'
                   f'块位置 ({xy[0]:.2f},{xy[1]:.2f})'))
    return out


def vlm_t11_verdict(frames, ffps, c, meta, rubric):
    """T11 专用裁决：整帧拼图 MLLM 裸看接近随机（Artifact-Bench）——
    改为同一区域连续 4 帧的放大裁剪并排，把证据推到脸上。"""
    import cv2
    import vlm_common as V
    H, W = frames[0].shape[:2]
    f0, f1 = c['film_frames']
    f1 = min(f1, len(frames) - 1)
    if 'region_640x360' in c:
        x0, y0, x1, y1 = c['region_640x360']
        sx, sy = W / 640.0, H / 360.0
        box = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
    else:
        cx, cy = c.get('region_xy_norm', [0.5, 0.5])
        box = [(cx - 0.22) * W, (cy - 0.22) * H, (cx + 0.22) * W, (cy + 0.22) * H]
    # padding ×1.4 并裁边
    bw, bh = box[2] - box[0], box[3] - box[1]
    box = [max(0, int(box[0] - 0.2 * bw)), max(0, int(box[1] - 0.2 * bh)),
           min(W, int(box[2] + 0.2 * bw)), min(H, int(box[3] + 0.2 * bh))]
    if box[2] - box[0] < 24 or box[3] - box[1] < 24:
        box = [0, 0, W, H]
    ids = sorted(set([f0, min(f0 + 1, f1), max(f1 - 1, f0), f1]))
    tiles = []
    for fi in ids:
        crop = frames[fi][box[1]:box[3], box[0]:box[2]]
        th = 300
        tile = cv2.resize(crop, (max(60, int(th * crop.shape[1] / crop.shape[0])), th))
        cv2.rectangle(tile, (0, 0), (92, 22), (0, 0, 0), -1)
        cv2.putText(tile, f'{fi / ffps:.3f}s', (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        tiles.append(tile)
    hmax = max(t.shape[0] for t in tiles)
    strip = np.hstack([np.pad(t, ((0, hmax - t.shape[0]), (0, 6), (0, 0)),
                              constant_values=24) for t in tiles])
    subj = f'subject "{c["subject"]}"' if c.get('subject') else 'the local region'
    rub = '\n'.join(f'- {x}' for x in rubric)
    p = (f'You are inspecting an AI-generated video for LOCAL temporal incoherence '
         f'(object-level glitch between adjacent frames). Intended idea: '
         f'"{meta["idea"]}"\nPre-registered criteria:\n{rub}\n\n'
         f'A localized detector flagged {subj} at t={c["start_s"]}-{c["end_s"]}s: '
         f'{c["signal"]}. Below are ENLARGED CROPS of the same region from 4 '
         f'consecutive/near frames (timestamps burned in).\n'
         f'Judge SPECIFICALLY: between adjacent crops, does the object teleport, '
         f'morph, swap limbs/parts, or change shape in a physically impossible way? '
         f'Normal motion blur / fast-but-continuous motion is acceptable.\n'
         f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
         f'"severity": 1-5, "reason": "<简体中文简述>", "confidence": 0.0-1.0}}')
    parsed, raw = ask_claude([{'text': p}, V.img_block(strip)])
    return parsed or {'verdict': 'PARSE_FAIL', 'reason': raw[:120]}


# ------------------------------------------------- V2 第一批：T17/T19/T20
def detect_t17(sig, fps, spans, shots, ex):
    """T17 运动动态性：幅度失配直判 + 重复循环候选（→VLM）。
    校准（10 成片实测）：motion=high 分镜帧差中位 12.1、正常最低 3.1；
    唯一 <2 的 (1.4) 是导演要求跃起接盘但柯基只小跑的真缺陷。"""
    n = len(sig['luminance'])
    d1 = align_n(sig['diff_d1'], n)
    F, cands = [], []
    AMP_TH = {'high': 2.0, 'medium': 0.5}
    for k, (a, b) in enumerate(spans):
        s = shots[min(k, len(shots) - 1)]
        lo, hi = int(a * fps), min(n, int(b * fps))
        seg = d1[lo:hi]
        seg = seg[~ex[lo:hi]] if ex[lo:hi].any() else seg    # 排除转场帧
        if len(seg) < int(fps):
            continue
        med = float(np.median(seg))
        lvl = s.get('motion_level', 'medium')
        if lvl in AMP_TH and med < AMP_TH[lvl]:
            F.append(dict(type='T17_motion_dynamics', start_s=round(a, 2),
                          end_s=round(b, 2), severity=3,
                          evidence=f'分镜{k + 1} 要求 motion={lvl} 但实际运动量极低'
                                   f'（帧差中位 {med:.2f}，同级正常 ≥{AMP_TH[lvl]}；'
                                   f'画面有像素变化但运动语义不达标，区别于 T3 硬冻结）',
                          confidence=0.85, verdict_by='detector'))
            continue
        # 重复循环候选：镜头内帧差曲线自相关周期峰（AIGV 不自然时间自相似性,ATSS）
        x = seg - seg.mean()
        if len(x) >= int(3 * fps) and x.std() > 0.3:
            ac = np.correlate(x, x, 'full')[len(x) - 1:]
            ac /= (ac[0] + 1e-9)
            lags = ac[int(0.5 * fps):int(min(2.5 * fps, len(ac) - 1))]
            if len(lags) and lags.max() > 0.85:
                cands.append(dict(shot=k + 1, start_s=round(a, 2), end_s=round(b, 2),
                                  peak=round(float(lags.max()), 2)))
    return F, cands[:1]          # 循环候选每片最多 1 个交 VLM


def detect_t20(meta, model, gate, clips_gate, trans, sig, fps, n):
    """T20 管线执行缺陷：规格对照 + 转场执行质量（近零成本，不看语义）。
    DirectorBench：转场质量是全行业最弱环节（均分 0.256）。"""
    from common import MODELS
    F = []
    shots = meta['shots']
    # 1) 规格对照：成片时长 = Σ分镜 − Σ交叠
    n_fade = sum(1 for t in trans if t['type'] in ('fade', 'dissolve'))
    expect_dur = sum(s['duration_s'] for s in shots) - 0.5 * n_fade
    if gate.get('duration_s') and abs(gate['duration_s'] - expect_dur) > 1.0:
        F.append(dict(type='T20_pipeline', start_s=0.0, end_s=round(gate['duration_s'], 2),
                      severity=3,
                      evidence=f'成片时长 {gate["duration_s"]:.1f}s 与分镜规划 '
                               f'{expect_dur:.1f}s 偏差 >1s（规格不符）',
                      confidence=0.95, verdict_by='detector'))
    want_fps = MODELS[model]['fps']
    if gate.get('fps') and abs(gate['fps'] - want_fps) > 1:
        F.append(dict(type='T20_pipeline', start_s=0.0, end_s=0.0, severity=3,
                      evidence=f'成片帧率 {gate["fps"]} 与规格 {want_fps} 不符',
                      confidence=0.95, verdict_by='detector'))
    for name, g in clips_gate.items():
        if not g.get('decodable'):
            F.append(dict(type='T20_pipeline', start_s=0.0, end_s=0.0, severity=5,
                          evidence=f'分镜 clip {name} 生成失败或无法解码',
                          confidence=0.99, verdict_by='detector'))
    # 2) 转场执行：dissolve/fade 窗内不应有硬切（有 = 未平滑执行/残影）
    d1 = align_n(sig['diff_d1'], n)
    for t in trans:
        lo, hi = int(t['start_s'] * fps), min(n, int(t['end_s'] * fps) + 1)
        if t['type'] in ('fade', 'dissolve') and hi > lo:
            if float(d1[lo:hi].max()) > 25.0:
                F.append(dict(type='T20_pipeline', start_s=t['start_s'],
                              end_s=t['end_s'], severity=3,
                              evidence=f'{t["type"]} 转场窗内出现帧差 '
                                       f'{d1[lo:hi].max():.0f}>25 的硬跳变'
                                       f'（转场未平滑执行/残影）',
                              confidence=0.85, verdict_by='detector'))
    return F


def check_rewrite_fidelity(meta, pdir):
    """T20 子项：导演改写忠实度 —— 创意关键要素是否在 wan_prompt 中丢失。
    纯文本 LLM diff，跨模型共享缓存（文献空白项，MovieAgent 只有间接指标）。"""
    cache = os.path.join(pdir, 'rewrite_check.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    shots_txt = '\n'.join(f'Shot {s["shot_id"]}: {s["wan_prompt"]}'
                          for s in meta['shots'])
    p = (f'用户创意（中文）："{meta["idea"]}"\n'
         f'导演 Agent 改写出的分镜 prompts：\n{shots_txt}\n\n'
         f'列出创意中的关键要素（主体/动作/场景/风格/文字内容），逐个判断是否被'
         f'至少一个分镜 prompt 覆盖。只报告确实完全丢失的关键要素（宽松判定，'
         f'语义等价即算覆盖）。用简体中文。\n'
         f'Respond ONLY with JSON: {{"missing_critical": ["<丢失的关键要素>"], '
         f'"note": "<一句话>"}}')
    parsed, _ = ask_claude([{'text': p}])
    out = parsed or {'missing_critical': []}
    json.dump(out, open(cache, 'w'), ensure_ascii=False)
    return out


def vlm_t19_cross_shot(frames, ffps, spans, meta, rubric, hue_ctx=''):
    """T19 跨镜头一致性：每镜头取中点帧并排 → 封闭式问题（角色/道具/风格跨镜头）。
    主流评估的盲区（Movie Gen 的 consistency 仍是单镜头内定义）。"""
    import cv2
    import vlm_common as V
    tiles = []
    for k, (a, b) in enumerate(spans):
        fi = min(len(frames) - 1, int((a + b) / 2 * ffps))
        t = frames[fi]
        th = 300
        tile = cv2.resize(t, (int(th * t.shape[1] / t.shape[0]), th))
        cv2.rectangle(tile, (0, 0), (110, 24), (0, 0, 0), -1)
        cv2.putText(tile, f'Shot {k + 1}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(tile)
    strip = np.hstack([np.pad(t, ((0, 0), (0, 8), (0, 0)), constant_values=24)
                       for t in tiles])
    shots_txt = '\n'.join(f'Shot {s["shot_id"]}: {s["wan_prompt"][:150]}'
                          for s in meta['shots'])
    p = (f'You are checking CROSS-SHOT consistency of a storyboard-generated video. '
         f'Storyboard:\n{shots_txt}\n{hue_ctx}\n'
         f'Below: one representative frame per shot, side by side.\n'
         f'Judge ONLY cross-shot issues: (1) is the main character/subject the SAME '
         f'entity across shots (clothing, colors, props, breed/model)? '
         f'(2) is the visual style/color-grade consistent? '
         f'(3) any prop that must persist (per storyboard) changed or vanished '
         f'between shots? Shot-to-shot camera/framing changes are expected.\n'
         f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", "severity": 1-5, '
         f'"aspect": "identity|style|prop|none", "reason": "<简体中文简述>", '
         f'"confidence": 0.0-1.0}}')
    parsed, raw = ask_claude([{'text': p}, V.img_block(strip)])
    return parsed or {'verdict': 'PARSE_FAIL', 'reason': raw[:120]}


def vlm_t17_loop(frames, ffps, c, meta, rubric):
    import vlm_common as V
    a, b = c['start_s'], c['end_s']
    ids = [min(len(frames) - 1, int((a + (b - a) * i / 9) * ffps)) for i in range(9)]
    sheet = V.contact_sheet([(frames[k], k / ffps) for k in ids], cols=3, tile_w=280)
    p = (f'Shot {c["shot"]} of an AI video was flagged for possible content LOOPING '
         f'(autocorrelation peak {c["peak"]}). Below are 9 evenly spaced frames.\n'
         f'Is the content unnaturally repeating/cycling (same motion pattern looping), '
         f'or is it legitimate periodic motion (walking, waves) / normal progression?\n'
         f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", "severity": 1-5, '
         f'"reason": "<简体中文简述>", "confidence": 0.0-1.0}}')
    parsed, raw = ask_claude([{'text': p}, V.img_block(sheet)])
    return parsed or {'verdict': 'PARSE_FAIL', 'reason': raw[:120]}


def crop_strip(frames, ffps, items, th=280):
    """items: [(t_s, box640_or_None, label)] → 标注时间戳的横向并排放大裁剪图。"""
    import cv2
    H, W = frames[0].shape[:2]
    sx, sy = W / 640.0, H / 360.0
    tiles = []
    for t_s, box, label in items:
        fi = min(len(frames) - 1, max(0, int(t_s * ffps)))
        f = frames[fi]
        if box:
            x0, y0 = int(box[0] * sx), int(box[1] * sy)
            x1, y1 = int(box[2] * sx), int(box[3] * sy)
            bw, bh = x1 - x0, y1 - y0
            x0, y0 = max(0, int(x0 - 0.2 * bw)), max(0, int(y0 - 0.2 * bh))
            x1, y1 = min(W, int(x1 + 0.2 * bw)), min(H, int(y1 + 0.2 * bh))
            crop = f[y0:y1, x0:x1] if (x1 - x0 >= 24 and y1 - y0 >= 24) else f
        else:
            crop = f
        tile = cv2.resize(crop, (max(80, int(th * crop.shape[1] / crop.shape[0])), th))
        cv2.rectangle(tile, (0, 0), (152, 24), (0, 0, 0), -1)
        cv2.putText(tile, f'{label} {t_s:.2f}s', (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(tile)
    hm = max(t.shape[0] for t in tiles)
    return np.hstack([np.pad(t, ((0, hm - t.shape[0]), (0, 6), (0, 0)),
                             constant_values=24) for t in tiles])


# ---------------------------------------------------------------- S4/S5 VLM
def make_rubric(meta):
    shots_txt = '\n'.join(f'Shot {s["shot_id"]}: {s["wan_prompt"]}' for s in meta['shots'])
    p = ('You are designing evaluation criteria for an AI-generated multi-shot video '
         'BEFORE seeing it (to avoid being biased by the output). The storyboard:\n'
         f'{shots_txt}\n\nWrite 4-6 hard pass/fail criteria focused on: subject identity '
         'consistency within each shot AND ACROSS shots (character clothing/props/'
         'style must persist from shot to shot), subject visibility, temporal '
         'continuity inside a shot (no unexpected cuts/jumps), and prompt adherence. '
         'One criterion MUST cover local frame-to-frame continuity: 同一对象相邻帧之间'
         '位置/形态/姿态必须连续，突变且无运动模糊解释则 FAIL。'
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


def vlm_window_verdict(frames, fps, win, meta, rubric, context, trans_txt=''):
    n = len(frames)
    c = int((win['start_s'] + win['end_s']) / 2 * fps)
    lo = max(0, min(n - 8, c - 4))
    step = max(1, int((win['end_s'] - win['start_s']) * fps / 8)) or 1
    ids = [min(n - 1, lo + i * step) for i in range(8)]
    rub = '\n'.join(f'- {x}' for x in rubric)
    p = (f'You are inspecting an AI-generated video assembled from a storyboard. '
         f'{trans_txt}Intended idea: "{meta["idea"]}"\n'
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
              'T11_local_incoherence': 'temporal', 'T17_motion_dynamics': 'temporal',
              'T19_cross_shot': 'subject', 'T20_pipeline': 'structural',
              'T12_anatomy': 'subject', 'T13_physics': 'semantic',
              'T14_interaction': 'subject', 'T15_text': 'semantic',
              'T16_camera_motion': 'semantic',
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

    # S2 direct detectors（v1 全帧时序 + v2 第一批：T20 管线规格/转场执行、T17 幅度失配）
    t0 = time.time()
    findings = detect_hard(sc['signals'], sc['cuts_frames'], fps, meta, model, spans, trans)
    findings += detect_t20(meta, model, gate, clips_gate, trans, sc['signals'], fps, n)
    ex_all = exempt_mask(n, fps, trans, TH['trans_margin_s'])
    t17_f, t17_loop_cands = detect_t17(sc['signals'], fps, spans, meta['shots'], ex_all)
    findings += t17_f
    timing['s2_detect_s'] = round(time.time() - t0, 2)

    # S3 subject probes
    t0 = time.time()
    try:
        sub_findings, sub_info = subject_probes(film, meta, model, spans, fps, device)
    except Exception as e:
        sub_findings, sub_info = [], {'error': str(e)[:300]}
    findings = merge_findings(findings + sub_findings)
    timing['s3_probes_s'] = round(time.time() - t0, 2)

    # S3b v2 第二批探针：T12 人体关键点 / T14 交互区域 / T16 相机运动 / T15 文字
    t0 = time.time()
    import probes_v2 as PV2
    person_kw = ('person', 'man', 'woman', 'soldier', 'pedestrian', 'fencer',
                 'player', 'elderly', 'people', 'figure', 'dog', 'corgi')
    has_person = any(any(k in x.lower() for k in person_kw[:10])
                     for s in meta['shots'] for x in s.get('expected_subjects', []))
    t12_cands, hand_regs = [], []
    kfps = None
    if has_person:
        try:
            sub_frames = decode_sub(film)
            kps = PV2.person_keypoints(sub_frames[::2], device)
            kfps = (fps / SUB_EVERY) / 2
            t12_cands = PV2.t12_bone_stats(kps, kfps, spans, meta['shots'])
            hand_regs = PV2.wrist_regions(kps, kfps, spans, 640, 360)
        except Exception as e:
            print('[v2] keypoint probe fail:', str(e)[:120])
    inter_regs = PV2.interaction_regions(sub_info, spans,
                                         sub_info.get('sub_fps') or fps / 3, meta)
    findings += PV2.t16_camera_check(sc.get('camera_flow', {}), fps, spans,
                                     meta['shots'], None)
    text_shots = [(k, s, PV2.prompt_texts(s)) for k, s in enumerate(meta['shots'])
                  if k < len(spans) and PV2.prompt_texts(s)]
    import re as _re
    PHYS = _re.compile(r'bounc|fall|drop|pour|splash|roll|slide|toss|throw|'
                       r'gravity|ripple|settle|rebound')
    phys_shots = [(k, s) for k, s in enumerate(meta['shots'])
                  if k < len(spans) and PHYS.search(s['wan_prompt'].lower())][:2]
    timing['s3b_v2_s'] = round(time.time() - t0, 2)

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
        # 计划转场时间表：VLM 裁决必须知道镜头边界在哪，否则会把计划切点
        # 误判为"镜头内时间连续性断裂"（p4 实测盲区）
        trans_txt = ''
        if trans:
            items = ', '.join(f'{t["type"]} at {t["start_s"]}-{t["end_s"]}s'
                              for t in trans)
            trans_txt = (f'PLANNED shot transitions (composition/scene changes at these '
                         f'times are EXPECTED, not defects): {items}. Shot boundaries: '
                         + ', '.join(f'shot{i + 1}={a:.1f}-{b:.1f}s'
                                     for i, (a, b) in enumerate(spans)) + '. ')

        def window_presence(w):
            """窗口内各主体的跟踪检出率（像素证据，用于约束/否决 VLM 印象）。"""
            if not presence or not p_sub_fps:
                return {}
            lo = max(0, int(w['start_s'] * p_sub_fps))
            hi = max(lo + 1, int(w['end_s'] * p_sub_fps))
            return {s: sum(arr[lo:hi]) / max(1, len(arr[lo:hi]))
                    for s, arr in presence.items() if arr[lo:hi]}

        def judge_window(w):
            """融合候选窗裁决。主体消失类结论采用双重验证（GroundingDINO 主判）：
            - VLM 声称主体消失 + 跟踪器同窗口确有缺失 → 双重确认，置信扣分
            - VLM 声称消失但单实例主体跟踪检出率满格 → 以跟踪器为准，否决 VLM
            - 多实例主体（"two fencers"）跟踪器只能证明"至少一个在场"，保留 VLM 单票但降置信"""
            ctx = f'fused anomaly score {w["score"]} (soft signals)'
            v = vlm_window_verdict(frames, ffps, w, meta, rubric, ctx, trans_txt)
            if v.get('verdict') != 'defect':
                return None
            claim = (str(v.get('type', '')) + str(v.get('reason', ''))).lower()
            vanish_kw = ['消失', '不见', 'disappear', 'vanish', 'missing',
                         'absent', 'discontinu']
            conf = float(v.get('confidence', 0.6))
            by = 'vlm'
            if any(k in claim for k in vanish_kw):
                wp = window_presence(w)
                singular = {s: r for s, r in wp.items()
                            if s.lower().startswith(('a ', 'an '))}
                if singular and min(singular.values()) >= 0.95:
                    vlm_rejected.append(dict(
                        window=[w['start_s'], w['end_s']],
                        vlm_claim=str(v.get('reason', ''))[:150],
                        rejected_because='跟踪器主判否决：窗口内单实例主体检出率 '
                        + ', '.join(f'{s}={r * 100:.0f}%' for s, r in singular.items())))
                    return None
                if wp and min(wp.values()) < 0.6:
                    by = 'dual'                 # VLM + 跟踪器双重确认
                    conf = max(conf, 0.9)
                else:
                    conf = min(conf, 0.6)       # 跟踪器无法佐证，VLM 单票降置信
            return dict(type='vlm_defect', start_s=w['start_s'], end_s=w['end_s'],
                        severity=int(v.get('severity', 3)),
                        evidence=f'{v.get("type")}: {v.get("reason", "")}'[:200],
                        confidence=conf, verdict_by=by)

        def judge_candidate(c):
            ftype, tpl = CAND_MAP[c['type']]
            win = {'start_s': c['start_s'], 'end_s': max(c['end_s'], c['start_s'] + 0.5)}
            ctx = tpl.format(**{k: c.get(k, '') for k in ('subject', 'shot', 'signal')})
            v = vlm_window_verdict(frames, ffps, win, meta, rubric, ctx, trans_txt)
            if v.get('verdict') == 'defect':
                # T5 由跟踪器缺失发起 + VLM 确认意图 → 天然双重验证；T7/T8 单票 VLM
                by = 'dual' if ftype == 'T5_out_of_frame' else 'vlm'
                return dict(type=ftype, start_s=c['start_s'], end_s=c['end_s'],
                            severity=int(v.get('severity', 3)),
                            evidence=f'{c["subject"]}（分镜{c["shot"]}，'
                                     f'{c.get("signal", "检测缺失")}）：'
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.6)), verdict_by=by)

        def judge_t11(c):
            v = vlm_t11_verdict(frames, ffps, c, meta, rubric)
            if v.get('verdict') == 'defect':
                who = c.get('subject') or '局部区域'
                return dict(type='T11_local_incoherence', start_s=c['start_s'],
                            end_s=c['end_s'], severity=int(v.get('severity', 3)),
                            evidence=f'{who}（分镜{c["shot"]}，{c["signal"]}）：'
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='dual')     # 局部信号 + VLM 双重确认

        def judge_t10(k_s):
            k, s = k_s
            r = vlm_t10_alignment(frames, ffps, spans[k], s, rubric)
            return {'shot_id': s['shot_id'],
                    **{kk: r.get(kk) for kk in ('score', 'missing', 'reason')}}

        T10_VOTES = 3        # Bedrock 图像输入非严格确定，边界 2/3 分会翻转 → 三票取中位

        # T11 候选：信号 A（块残差，全局低局部高）+ 信号 C（主体轨迹，已在探针中发起）
        all_cands = sub_info.get('candidates', [])
        t11_cands = [c for c in all_cands if c['type'] == 'T11_local_candidate']
        other_cands = [c for c in all_cands if c['type'] != 'T11_local_candidate']
        ex_mask = exempt_mask(n, fps, trans, TH['trans_margin_s'])
        t11_cands += detect_t11_blocks(sc, fps, ex_mask, sc['cuts_frames'],
                                       spans, meta['shots'])

        def in_exempt(c):
            """转场豁免也适用于轨迹信号：dissolve 叠化会出现双主体、cut 处构图跳变，
            都是预期转场效果而非 T11。"""
            lo = int(c['start_s'] * fps)
            hi = min(n - 1, int(c['end_s'] * fps))
            return bool(ex_mask[lo:hi + 1].any())

        t11_cands = [c for c in t11_cands if not in_exempt(c)]

        def judge_t19():
            # 主体跨镜头色相直方图距离作为附加像素证据（>0.5 提示 VLM 重点核对）
            hue_flags = []
            for subj, per_shot in sub_info.get('shot_hue', {}).items():
                ks = sorted(per_shot)
                if len(ks) < 2:
                    continue
                ref = per_shot[ks[0]]
                for kk in ks[1:]:
                    h = per_shot[kk]
                    d = 1 - float(np.dot(h, ref)) / (
                        np.linalg.norm(h) * np.linalg.norm(ref) + 1e-8)
                    if d > 0.5:
                        hue_flags.append(f'"{subj}" shot{ks[0]}→shot{kk} '
                                         f'hue-hist dist {d:.2f}')
            hue_ctx = ('A color probe flagged cross-shot appearance drift: '
                       + '; '.join(hue_flags) + '.\n') if hue_flags else ''
            v = vlm_t19_cross_shot(frames, ffps, spans, meta, rubric, hue_ctx)
            if v.get('verdict') == 'defect':
                return dict(type='T19_cross_shot', start_s=0.0,
                            end_s=round(spans[-1][1], 2),
                            severity=int(v.get('severity', 3)),
                            evidence=f'跨镜头{v.get("aspect", "")}不一致'
                                     f'{("（色相探针同报：" + "; ".join(hue_flags) + "）") if hue_flags else ""}：'
                                     f'{v.get("reason", "")}'[:250],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='dual' if hue_flags else 'vlm')

        def judge_loop(c):
            v = vlm_t17_loop(frames, ffps, c, meta, rubric)
            if v.get('verdict') == 'defect':
                return dict(type='T17_motion_dynamics', start_s=c['start_s'],
                            end_s=c['end_s'], severity=int(v.get('severity', 3)),
                            evidence=f'分镜{c["shot"]} 内容不自然循环（自相关峰 '
                                     f'{c["peak"]}）：{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='dual')

        import vlm_common as V

        def judge_t12_hands():
            """手部/持物区域批量核查（一张并排图一次调用）—— p4 类雨伞/手缺陷主打。"""
            if not hand_regs:
                return []
            items = [(t, box, f'H{i + 1}') for i, (t, box, sh)
                     in enumerate(hand_regs[:6])]
            strip = crop_strip(frames, ffps, items)
            p = (f'You are inspecting HAND/GRIP regions of an AI-generated video '
                 f'(idea: "{meta["idea"]}"). Each labeled crop (H1..H{len(items)}) is an '
                 f'enlarged wrist/hand region, possibly holding a prop.\n'
                 f'For EACH crop judge: finger count/anatomy normal? hand merged/'
                 f'broken? grip physically plausible (prop floating with no hand '
                 f'contact, hand passing through handle, duplicated handles)?\n'
                 f'STRICT RULE: report ONLY clearly VISIBLE structural anomalies: '
                 f'wrong finger count, fused/broken hand, prop floating with no '
                 f'contact, duplicated/extra handles or hooks, hand passing through '
                 f'object. For FINGER DETAILS only: silhouette/DoF/motion blur where '
                 f'fingers simply cannot be seen is NOT a defect — but object-level '
                 f'structure (double handle, detached handle, floating prop) IS '
                 f'judgeable even in silhouette.\n'
                 f'Respond ONLY with JSON: {{"abnormal": [{{"id": "H1", '
                 f'"severity": 1-5, "issue": "<简体中文简述>"}}]}} '
                 f'(empty list if all normal)')
            parsed, _ = ask_claude([{'text': p}, V.img_block(strip)])
            out = []
            for ab in (parsed or {}).get('abnormal', []):
                try:
                    i = int(str(ab.get('id', 'H1'))[1:]) - 1
                    t_s, _, sh = hand_regs[i]
                except (ValueError, IndexError):
                    continue
                out.append(dict(type='T12_anatomy', start_s=t_s,
                                end_s=round(t_s + 0.5, 2),
                                severity=int(ab.get('severity', 3)),
                                evidence=f'手部/持物核查（分镜{sh}）：'
                                         f'{ab.get("issue", "")}'[:200],
                                confidence=0.8, verdict_by='vlm'))
            return out

        def judge_t12_bone(c):
            a, b = c['start_s'], c['end_s']
            items = [(a + (b - a) * f, None, 'ABC'[i])
                     for i, f in enumerate((0.2, 0.5, 0.8))]
            strip = crop_strip(frames, ffps, items)
            p = (f'A pose probe flagged shot {c["shot"]}: {c["signal"]}. Below are 3 '
                 f'frames (A/B/C). Judge: do the person\'s limbs stay anatomically '
                 f'consistent (no limb morphing/stretching/extra limbs) across frames? '
                 f'Perspective changes are acceptable.\n'
                 f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
                 f'"severity": 1-5, "reason": "<简体中文>", "confidence": 0.0-1.0}}')
            parsed, _ = ask_claude([{'text': p}, V.img_block(strip)])
            v = parsed or {}
            if v.get('verdict') == 'defect':
                return dict(type='T12_anatomy', start_s=a, end_s=b,
                            severity=int(v.get('severity', 3)),
                            evidence=f'人体骨长变异（分镜{c["shot"]}，{c["signal"]}）：'
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='dual')

        def judge_t14(r):
            items = [(max(0, r['t_s'] + d), r['box640'], 'ABC'[i])
                     for i, d in enumerate((-0.4, 0.0, 0.4))]
            strip = crop_strip(frames, ffps, items)
            p = (f'Inspect the INTERACTION between "{r["person"]}" and "{r["prop"]}" '
                 f'in shot {r["shot"]} of an AI video (3 enlarged crops A/B/C, ~0.4s '
                 f'apart). Judge ONLY physical interaction plausibility: is the prop '
                 f'held/supported by a hand with real contact? any floating prop that '
                 f'follows the person with NO contact, hand passing through the prop, '
                 f'duplicated/broken handles, or contact at an impossible point?\n'
                 f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
                 f'"severity": 1-5, "reason": "<简体中文>", "confidence": 0.0-1.0}}')
            parsed, _ = ask_claude([{'text': p}, V.img_block(strip)])
            v = parsed or {}
            if v.get('verdict') == 'defect':
                return dict(type='T14_interaction', start_s=round(r['t_s'] - 0.4, 2),
                            end_s=round(r['t_s'] + 0.4, 2),
                            severity=int(v.get('severity', 3)),
                            evidence=f'{r["person"]}×{r["prop"]} 交互（分镜{r["shot"]}）：'
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='dual')

        def judge_t15(k, s, texts):
            a, b = spans[k]
            sign_box_key = next((x for x in sub_info.get('boxes', {})
                                 if any(w in x.lower() for w in
                                        ('sign', 'text', 'label', 'logo'))), None)
            bxs = sub_info.get('boxes', {}).get(sign_box_key, [])
            sfps = sub_info.get('sub_fps') or fps / 3
            items = []
            for i, f in enumerate((0.25, 0.5, 0.8)):
                t_s = a + (b - a) * f
                bx = bxs[min(len(bxs) - 1, int(t_s * sfps))] if bxs else None
                items.append((t_s, bx, 'ABC'[i]))
            strip = crop_strip(frames, ffps, items)
            p = (f'Shot {k + 1} requires on-screen text: {texts}. Below are 3 frames '
                 f'(A/B/C) with the text region enlarged when detected.\n'
                 f'(1) Read the actual visible text in each frame. (2) Is it exactly '
                 f'the required text (no missing/garbled letters)? (3) Is it stable '
                 f'across frames (no morphing/drifting glyphs)?\n'
                 f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
                 f'"read": ["<各帧读到的>"], "severity": 1-5, '
                 f'"reason": "<简体中文>", "confidence": 0.0-1.0}}')
            parsed, _ = ask_claude([{'text': p}, V.img_block(strip)])
            v = parsed or {}
            if v.get('verdict') == 'defect':
                return dict(type='T15_text', start_s=round(a, 2), end_s=round(b, 2),
                            severity=int(v.get('severity', 3)),
                            evidence=f'画面文字应为 {texts}，实际 {v.get("read", "?")}'
                                     f'（分镜{k + 1}）：{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='vlm')

        def judge_t13(k, s):
            a, b = spans[k]
            n_f = len(frames)
            ids = [min(n_f - 1, int((a + (b - a) * (i + 0.5) / 8) * ffps))
                   for i in range(8)]
            sheet = V.contact_sheet([(frames[j], j / ffps) for j in ids],
                                    cols=4, tile_w=300)
            p = (f'Physics check for shot {k + 1} of an AI video.\n'
                 f'Shot prompt: "{s["wan_prompt"]}"\n'
                 f'Step 1: state 2-3 physics assertions this scene must satisfy '
                 f'(gravity/momentum/contact response/state persistence, e.g. "a '
                 f'bouncing ball\'s rebound height must decrease").\n'
                 f'Step 2: check each against the 8 frames (timestamps burned in).\n'
                 f'Respond ONLY with JSON: {{"verdict": "defect|acceptable", '
                 f'"violated": ["<被违反的断言,中文>"], "severity": 1-5, '
                 f'"reason": "<简体中文>", "confidence": 0.0-1.0}}')
            parsed, _ = ask_claude([{'text': p}, V.img_block(sheet)])
            v = parsed or {}
            if v.get('verdict') == 'defect':
                return dict(type='T13_physics', start_s=round(a, 2), end_s=round(b, 2),
                            severity=int(v.get('severity', 3)),
                            evidence=f'物理断言违反（分镜{k + 1}）：'
                                     f'{"；".join(v.get("violated", [])[:3])} —— '
                                     f'{v.get("reason", "")}'[:220],
                            confidence=float(v.get('confidence', 0.7)),
                            verdict_by='vlm')

        # 除 rubric 外全部 VLM 调用并行（Bedrock 侧无依赖）
        with ThreadPoolExecutor(max_workers=16) as pool:
            fut_wins = [pool.submit(judge_window, w) for w in wins]
            fut_cands = [pool.submit(judge_candidate, c) for c in other_cands] \
                + [pool.submit(judge_t11, c) for c in t11_cands]
            fut_t10 = [(k, [pool.submit(judge_t10, (k, s)) for _ in range(T10_VOTES)])
                       for k, s in enumerate(meta['shots']) if k < len(spans)]
            # T19 跨镜头一致性（多分镜才有意义）；T17 循环候选；T20 改写忠实度
            fut_t19 = pool.submit(judge_t19) if len(spans) >= 2 else None
            fut_loops = [pool.submit(judge_loop, c) for c in t17_loop_cands]
            fut_rw = pool.submit(check_rewrite_fidelity, meta, pdir)
            # v2 第二/三批：T12 手部+骨长 / T14 交互 / T15 文字 / T13 物理
            fut_hands = pool.submit(judge_t12_hands) if hand_regs else None
            fut_bones = [pool.submit(judge_t12_bone, c) for c in t12_cands[:2]]
            fut_inter = [pool.submit(judge_t14, r) for r in inter_regs]
            fut_text = [pool.submit(judge_t15, k, s, tx) for k, s, tx in text_shots]
            fut_phys = [pool.submit(judge_t13, k, s) for k, s in phys_shots]
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
            for f in ([fut_t19] if fut_t19 else []) + fut_loops + fut_bones \
                    + fut_inter + fut_text + fut_phys:
                vlm_calls += 1
                r = f.result()
                if r:
                    findings.append(r)
            if fut_hands:
                vlm_calls += 1
                findings += fut_hands.result()
            rw = fut_rw.result()
            vlm_calls += 1
            if rw.get('missing_critical'):
                findings.append(dict(
                    type='T20_pipeline', start_s=0.0, end_s=0.0, severity=2,
                    evidence=f'导演改写丢失创意关键要素：'
                             f'{"、".join(rw["missing_critical"][:5])}'
                             f'（{rw.get("note", "")}）'[:220],
                    confidence=0.8, verdict_by='llm'))
        t10.sort(key=lambda x: x['shot_id'])
        timing['s4_vlm_s'] = round(time.time() - t0, 2)

    findings = merge_findings(findings)      # 终段合并（VLM 各通道可能报同窗同类）
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
