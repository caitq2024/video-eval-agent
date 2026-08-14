#!/usr/bin/env python3
"""GPU 帧处理基准（A100）：逐组件计时,与 CPU 实测对照。

用法: /opt/pytorch/bin/python3 gpu_bench.py <video> [sub_every]
输出: probes/gpu_bench_<name>.json
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

DEV = 'cuda:0'
FLOW_SIZE = (320, 184)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def bench(path, sub_every=3):
    name = os.path.basename(path)
    t = {}
    t0 = time.time()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lum, grays, sub = [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append(g)
        lum.append(float(g.mean()))
        if i % sub_every == 0:
            sub.append(cv2.resize(f, FLOW_SIZE))
        i += 1
    cap.release()
    n = len(grays)
    t['decode_s'] = round(time.time() - t0, 2)

    t0 = time.time()
    d1 = [0.0] + [float(np.abs(grays[k] - grays[k - 1]).mean()) for k in range(1, n)]
    la = np.asarray(lum)
    flick = np.abs(la[1:-1] - (la[:-2] + la[2:]) / 2)
    t['pixel_s'] = round(time.time() - t0, 2)

    # CLIP on GPU
    import open_clip
    from PIL import Image
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(DEV).eval()
    t['clip_load_s'] = round(time.time() - t0, 2)
    t0 = time.time()
    feats = []
    with torch.no_grad(), torch.autocast('cuda'):
        for k in range(0, len(sub), 256):
            batch = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in sub[k:k + 256]]).to(DEV)
            feats.append(model.encode_image(batch).float().cpu())
    torch.cuda.synchronize()
    t['clip_s'] = round(time.time() - t0, 2)

    # RAFT small on GPU, 批量 pair
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    t0 = time.time()
    raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEV).eval()
    t['raft_load_s'] = round(time.time() - t0, 2)
    t0 = time.time()
    tens = torch.stack([
        torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1
        for f in sub])
    warp_res = []
    B = 32
    h, w = FLOW_SIZE[1], FLOW_SIZE[0]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    with torch.no_grad():
        for k in range(0, len(sub) - 1, B):
            a = tens[k:k + B].to(DEV)
            b = tens[k + 1:k + 1 + B].to(DEV)
            m = min(a.shape[0], b.shape[0])
            flow = raft(a[:m], b[:m])[-1].cpu().numpy()
            for j in range(m):
                mx, my = gx + flow[j, 0], gy + flow[j, 1]
                warped = cv2.remap(sub[k + j + 1].astype(np.float32), mx, my,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                warp_res.append(float(np.abs(warped - sub[k + j].astype(np.float32)).mean()))
    torch.cuda.synchronize()
    t['raft_s'] = round(time.time() - t0, 2)

    t0 = time.time()
    from scenedetect import detect, ContentDetector
    cuts = detect(path, ContentDetector(threshold=27.0))
    t['scenedetect_s'] = round(time.time() - t0, 2)

    dur = n / fps
    total = t['decode_s'] + t['pixel_s'] + t['clip_s'] + t['raft_s'] + t['scenedetect_s']
    out = {'video': name, 'n_frames': n, 'duration_s': round(dur, 1),
           'sub_frames': len(sub), 'device': torch.cuda.get_device_name(0),
           'timing': t, 'total_compute_s': round(total, 2),
           'realtime_factor': round(total / dur, 2)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(BASE, 'probes', f'gpu_bench_{name.replace(".mp4", "")}.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    bench(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
