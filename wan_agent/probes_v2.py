#!/usr/bin/env python3
"""TAXONOMY v2 第二/三批探针：T12 人体解剖 / T14 交互失真 / T16 相机运镜 /
T15 画面文字 / T13 物理断言。

设计原则同 v1：探针宽召回定位候选区域，证据裁剪放大交 VLM 封闭式裁决；
禁止全帧均值；VLM 不裸看整帧。
- T12：torchvision KeypointRCNN（免装 mmpose）→ 骨长时间维变异 + 腕部手区裁剪
- T14：GroundingDINO 人/物 bbox 邻接 → 交互区域（握持/接触）裁剪
- T16：fast_scan 已产出的全局光流平移/散度 vs 分镜 camera 字段（只判高置信矛盾）
- T15：wan_prompt 含引号/全大写文字 → 牌匾区域裁剪
"""
import re

import numpy as np

COCO_BONES = [(5, 7), (7, 9), (6, 8), (8, 10),      # 双臂
              (11, 13), (13, 15), (12, 14), (14, 16),  # 双腿
              (5, 6), (11, 12), (5, 11), (6, 12)]       # 躯干
_KP_MODEL = {}


def get_kp_model(device):
    if device not in _KP_MODEL:
        import torch
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
        m = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
        _KP_MODEL[device] = m
    return _KP_MODEL[device]


def person_keypoints(frames_rgb, device, batch=8):
    """frames_rgb: N,H,W,3 uint8（探针子采样帧）→ 每帧最高分人物的 (17,3) 关键点或 None"""
    import torch
    m = get_kp_model(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(frames_rgb), batch):
            imgs = [torch.from_numpy(f.copy()).permute(2, 0, 1).float().div(255).to(device)
                    for f in frames_rgb[i:i + batch]]
            for r in m(imgs):
                if len(r['scores']) and float(r['scores'][0]) > 0.75:
                    kp = r['keypoints'][0].cpu().numpy()        # 17,3 (x,y,vis)
                    ks = r['keypoints_scores'][0].cpu().numpy()
                    kp[:, 2] = ks
                    out.append(kp)
                else:
                    out.append(None)
    return out


def t12_bone_stats(kps, sub_fps, spans, shots):
    """骨长时间维变异（骨头长度帧间不该变，HumanScore 谱系）→ 候选（分镜级）。"""
    cands = []
    for k, (a, b) in enumerate(spans):
        lo, hi = int(a * sub_fps), int(b * sub_fps)
        seg = [kp for kp in kps[lo:hi] if kp is not None]
        if len(seg) < 6:
            continue
        # 相对骨长（除以躯干对角线做尺度归一,只统计两端可见的骨）
        ratios = {bi: [] for bi in range(len(COCO_BONES))}
        for kp in seg:
            torso = np.hypot(kp[5, 0] - kp[12, 0], kp[5, 1] - kp[12, 1])
            if torso < 20:
                continue
            for bi, (p, q) in enumerate(COCO_BONES):
                if kp[p, 2] > 2 and kp[q, 2] > 2:
                    ratios[bi].append(
                        np.hypot(kp[p, 0] - kp[q, 0], kp[p, 1] - kp[q, 1]) / torso)
        worst = 0.0
        for bi, v in ratios.items():
            if len(v) >= 5:
                v = np.asarray(v)
                cv = float(v.std() / (v.mean() + 1e-6))
                worst = max(worst, cv)
        if worst > 0.35:          # 骨长变异 >35%（透视会带来一些,门槛放宽）
            cands.append(dict(shot=k + 1, start_s=round(a, 2), end_s=round(b, 2),
                              signal=f'人体骨长时间维变异 {worst:.2f}>0.35'
                                     f'（骨头长度帧间不应变,肢体 morphing 嫌疑）'))
    return cands


def wrist_regions(kps, sub_fps, spans, w, h):
    """腕部关键点 → 手部区域（含持物）。返回 [(t_s, box640)]，每分镜取中段 2 个时点。"""
    regs = []
    for k, (a, b) in enumerate(spans):
        for frac in (0.5, 0.85):
            i = int((a + (b - a) * frac) * sub_fps)
            if i >= len(kps) or kps[i] is None:
                continue
            kp = kps[i]
            scale = max(np.hypot(kp[5, 0] - kp[12, 0], kp[5, 1] - kp[12, 1]), 40)
            for wrist in (9, 10):
                if kp[wrist, 2] > 2:
                    x, y = kp[wrist, 0], kp[wrist, 1]
                    r = scale * 0.45
                    regs.append((round(i / sub_fps, 2),
                                 [max(0, x - r), max(0, y - r),
                                  min(w, x + r), min(h, y + r)], k + 1))
    return regs


def interaction_regions(sub_info, spans, sub_fps, meta):
    """T14：人物主体与道具主体 bbox 邻接/重叠 → 交互区域（握持点）。
    用探针已有的逐帧 bbox；每分镜取一个代表交互窗。"""
    pres = sub_info.get('presence', {})
    boxes_all = sub_info.get('boxes', {})
    if not boxes_all:
        return []
    person_kw = ('person', 'man', 'woman', 'soldier', 'pedestrian', 'fencer',
                 'player', 'elderly', 'people', 'figure')
    persons = [s for s in boxes_all if any(k in s.lower() for k in person_kw)]
    props = [s for s in boxes_all if s not in persons]
    regs = []
    for k, (a, b) in enumerate(spans):
        lo, hi = int(a * sub_fps), int(b * sub_fps)
        for pe in persons:
            for pr in props:
                best = None
                for i in range(lo, min(hi, len(boxes_all[pe]))):
                    bp, bq = boxes_all[pe][i], boxes_all[pr][i]
                    if bp is None or bq is None:
                        continue
                    # 相邻或重叠（gap < 道具对角线的 30%）
                    gap_x = max(bq[0] - bp[2], bp[0] - bq[2], 0)
                    gap_y = max(bq[1] - bp[3], bp[1] - bq[3], 0)
                    diag = np.hypot(bq[2] - bq[0], bq[3] - bq[1])
                    d = np.hypot(gap_x, gap_y)
                    if best is None or d < best[0]:
                        best = (d, i, bp, bq, diag)
                if best and best[0] < 0.3 * best[4]:
                    _, i, bp, bq, _ = best
                    u = [min(bp[0], bq[0]), min(bp[1], bq[1]),
                         max(bp[2], bq[2]), max(bp[3], bq[3])]
                    regs.append(dict(shot=k + 1, t_s=round(i / sub_fps, 2),
                                     person=pe, prop=pr,
                                     box640=[float(v) for v in u]))
    return regs[:3]


TEXT_RE = re.compile(r'"([^"]{2,30})"|\'([^\']{2,30})\'|\b([A-Z]{3,12})\b')


def prompt_texts(shot):
    """从 wan_prompt 提取应出现在画面里的文字（引号内 / 全大写词）。"""
    skips = {'ENGLISH', 'ONLY', 'JSON', 'CLOSE', 'WIDE'}
    out = []
    for m in TEXT_RE.finditer(shot.get('wan_prompt', '')):
        t = next(g for g in m.groups() if g)
        if t.upper() not in skips and not t.islower():
            out.append(t)
    return list(dict.fromkeys(out))[:2]


def t16_camera_check(cam, fps, spans, shots, ex):
    """T16：全局光流 vs 分镜 camera 字段。只判高置信矛盾（static 却持续平移）。
    dx/dy 单位 = 像素/子采样步（RAFT 320×184）。"""
    F = []
    dx = np.asarray(cam.get('dx', []), np.float32)
    dy = np.asarray(cam.get('dy', []), np.float32)
    if not len(dx):
        return F
    mag = np.hypot(dx, dy)
    for k, (a, b) in enumerate(spans):
        s = shots[min(k, len(shots) - 1)]
        lo, hi = int(a * fps / 3), int(b * fps / 3)     # warp 子采样时间轴
        seg = mag[max(1, lo):hi]
        if len(seg) < 5:
            continue
        med = float(np.median(seg))
        frac_moving = float((seg > 1.2).mean())
        if s.get('camera') == 'static' and med > 1.5 and frac_moving > 0.6:
            F.append(dict(type='T16_camera_motion', start_s=round(a, 2),
                          end_s=round(b, 2), severity=3,
                          evidence=f'分镜{k + 1} 要求 static 机位，但全局光流中位平移 '
                                   f'{med:.1f}px/步、{frac_moving * 100:.0f}% 时间在移动'
                                   f'（机位漂移/未按运镜要求执行）',
                          confidence=0.85, verdict_by='detector'))
    return F
