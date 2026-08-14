#!/usr/bin/env python3
"""T8/T9 探测器升级实验（GPU）。

T9 物体凭空消失：GroundingDINO(tiny) 开放词汇检测("an orange ball. a green square.")
  逐子采样帧检测 → 每个物体的检测存活轨迹;首秒存在的物体连续缺失 ≥2 帧 → vanish 直判。
  主体(ball)缺失不走 vanish——归 T5 出界通道(已有跟踪+VLM 裁决),避免类型混淆。
T8 身份漂移：ball 的 bbox crop 喂 DINOv2(vits14) 提特征,对首秒参考(中位数)算余弦距离;
  距离 > 阈值 且持续 ≥2 帧 → identity_swap 直判(不再依赖 VLM,颜色变化也能抓)。

用法: /opt/pytorch/bin/python3 t8_t9_probes.py
输出: results/t8_t9_report.json + stdout 摘要
"""
import json
import os
import subprocess
import time

import numpy as np
import torch

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FFMPEG = '/home/ec2-user/efs/agent_evaluation/video_eval/bin/ffmpeg'
DEV = os.environ.get('DEV', 'cuda:0')
SUB_EVERY = 3
PROMPT = 'an orange ball. a green square.'
DET_TH = 0.35          # 检测置信度
DRIFT_TH = 0.35        # DINO 余弦距离阈值(先验,随后用 clean 校准检查)
MISS_RUN = 2           # 连续缺失帧数(子采样时间轴)

VIDEOS = ['tax_swap_A', 'tax_swap_B', 'tax_vanish_A', 'tax_vanish_B', 'tax_clean',
          'tax_jump_A', 'tax_flicker_A', 'tax_deform_A', 'tax_crop_A', 'tax_black_A']


def decode_sub(path):
    """8fps 子采样帧(RGB uint8)"""
    cmd = [FFMPEG, '-loglevel', 'error', '-i', path,
           '-vf', f'select=not(mod(n\\,{SUB_EVERY})),scale=640:360', '-vsync', '0',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (640 * 360 * 3)
    return np.frombuffer(raw[:n * 640 * 360 * 3], np.uint8).reshape(n, 360, 640, 3)


def main():
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    gd = GroundingDinoForObjectDetection.from_pretrained(
        'IDEA-Research/grounding-dino-tiny').to(DEV).eval()
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEV).eval()

    report = {}
    for name in VIDEOS:
        t0 = time.time()
        frames = decode_sub(os.path.join(BASE, 'videos', name + '.mp4'))
        n = len(frames)
        # ---- GroundingDINO 批量检测 ----
        dets = {'ball': [None] * n, 'square': [None] * n}
        B = int(os.environ.get('DET_BATCH', 16))
        with torch.no_grad():
            for i in range(0, n, B):
                batch = [frames[j] for j in range(i, min(i + B, n))]
                inp = proc(images=batch, text=[PROMPT] * len(batch),
                           return_tensors='pt').to(DEV)
                out = gd(**inp)
                res = proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=DET_TH, text_threshold=0.25,
                    target_sizes=[(360, 640)] * len(batch))
                for k, r in enumerate(res):
                    for lb, sc, bx in zip(r['labels'], r['scores'], r['boxes']):
                        key = 'ball' if 'ball' in lb else ('square' if 'square' in lb else None)
                        if key and (dets[key][i + k] is None or sc > dets[key][i + k][0]):
                            dets[key][i + k] = (float(sc), [float(v) for v in bx])
        # ---- 实例关联: 用首秒锚点做轨迹关联(背景相似物不能顶替) ----
        def associate(key, max_jump=90):
            """返回每帧该实例的 bbox(或 None)。以首个检测为锚,按中心距离关联。"""
            track = [None] * n
            last_c = None
            for i in range(n):
                d = dets[key][i]
                if d is None:
                    continue
                bx = d[1]
                c = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                if last_c is None:
                    track[i] = bx
                    last_c = c
                elif abs(c[0] - last_c[0]) + abs(c[1] - last_c[1]) < max_jump:
                    track[i] = bx
                    last_c = c
                # 距离过远 = 背景相似物,不接受、也不更新锚
            return track

        ball_tr = associate('ball')
        sq_tr = associate('square', max_jump=60)   # 方块基本不动

        # ---- T8: ball crop → DINOv2 特征漂移 + 色相直方图漂移 ----
        def crop_of(i, bx):
            x0, y0, x1, y1 = [int(v) for v in bx]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(640, x1), min(360, y1)
            if x1 - x0 < 10 or y1 - y0 < 10:
                return None
            return frames[i][y0:y1, x0:x1]

        import cv2
        crops, hists, idxs = [], [], []
        for i in range(n):
            if ball_tr[i] is None:
                continue
            c = crop_of(i, ball_tr[i])
            if c is None:
                continue
            t = torch.from_numpy(c.copy()).permute(2, 0, 1).float() / 255.0
            crops.append(torch.nn.functional.interpolate(t.unsqueeze(0), (126, 126),
                                                         mode='bilinear'))
            hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                             [0, 180, 0, 256]).flatten()
            hists.append(h / (h.sum() + 1e-8))
            idxs.append(i)
        dino_drift = np.zeros(n)
        hist_drift = np.zeros(n)
        if crops:
            with torch.no_grad():
                feats = dino(torch.cat(crops).to(DEV))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            ref_n = max(1, sum(1 for i in idxs if i < 8))
            ref = feats[:ref_n].median(0).values
            ref = ref / ref.norm()
            dd = (1 - feats @ ref).cpu().numpy()
            H = np.stack(hists)
            href = np.median(H[:ref_n], axis=0)
            href = href / (href.sum() + 1e-8)
            hd = 1 - (H @ href) / (np.linalg.norm(H, axis=1) * np.linalg.norm(href) + 1e-8)
            for k, i in enumerate(idxs):
                dino_drift[i] = dd[k]
                hist_drift[i] = hd[k]
        drift = np.maximum(dino_drift, hist_drift)
        t8_events = []
        run = 0
        for i in range(n):
            if drift[i] > DRIFT_TH:
                run += 1
                if run == MISS_RUN:
                    t8_events.append(round((i - run + 1) * SUB_EVERY / 16, 2))
            else:
                run = 0
        # ---- T9: square 实例存活 ----
        sq_present = [sq_tr[i] is not None for i in range(n)]
        t9_events = []
        if any(sq_present[:8]):
            run = 0
            for i in range(8, n):
                if not sq_present[i]:
                    run += 1
                    if run == MISS_RUN + 1:      # 关联轨迹下连续缺失 3 帧(0.56s)
                        t9_events.append(round((i - run + 1) * SUB_EVERY / 16, 2))
                else:
                    run = 0
        ball_det_rate = sum(1 for b in ball_tr if b) / n
        sq_det_rate = sum(1 for b in sq_tr if b) / n
        report[name] = {
            'ball_track_rate': round(ball_det_rate, 2), 'square_track_rate': round(sq_det_rate, 2),
            'dino_drift_max': round(float(dino_drift.max()), 3),
            'hist_drift_max': round(float(hist_drift.max()), 3),
            't8_identity_swap': t8_events, 't9_square_vanish': t9_events,
            'elapsed_s': round(time.time() - t0, 1)}
        print(f"{name:14s} ball_tr={ball_det_rate:.2f} sq_tr={sq_det_rate:.2f} "
              f"dino={dino_drift.max():.3f} hist={hist_drift.max():.3f} "
              f"T8@{t8_events} T9@{t9_events}")
    json.dump(report, open(os.path.join(BASE, 'results', 't8_t9_report.json'), 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
