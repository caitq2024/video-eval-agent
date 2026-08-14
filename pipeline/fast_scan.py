#!/usr/bin/env python3
"""GPU 快速全帧扫描:单遍解码 + GPU 流水线。

相对 gpu_bench.py 的优化:
  1. ffmpeg 管道直接输出 480×270 rawvideo(多线程解码+缩放,免 1080p 全尺寸拷贝)
  2. 去掉 PySceneDetect 的第二次解码——用 HSV 通道帧差(ContentDetector 同款信号)自实现
  3. CLIP 预处理搬上 GPU(torch resize+normalize,不再逐帧 PIL)
  4. RAFT batch=64 + autocast fp16;warp 残差用 GPU grid_sample 向量化
  5. 解码线程与 GPU 计算重叠(边解边算)

用法: python3 fast_scan.py <video> [--device cuda:0]
输出: 每帧信号 + 耗时分解(JSON 到 stdout / probes/fast_<name>.json)
"""
import argparse
import json
import os
import subprocess
import threading
import time
from queue import Queue

import numpy as np
import torch

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FFMPEG = '/home/ec2-user/efs/agent_evaluation/video_eval/bin/ffmpeg'
DW, DH = 480, 270          # 解码分辨率(像素统计/切换检测)
FW, FH = 320, 184          # RAFT 分辨率
SUB_EVERY = 3              # CLIP/RAFT 子采样步长
CHUNK = 250


def probe_meta(path):
    out = subprocess.run([FFMPEG, '-i', path], capture_output=True, text=True).stderr
    import re
    m = re.search(r'(\d+(?:\.\d+)?) fps', out)
    fps = float(m.group(1)) if m else 25.0
    return fps


def decoder(path, q):
    """ffmpeg 管道解码线程:产出 (idx, bgr480) 块"""
    cmd = [FFMPEG, '-loglevel', 'error', '-i', path,
           '-vf', f'scale={DW}:{DH}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=DW * DH * 3 * 8)
    idx = 0
    buf_frames = []
    nbytes = DW * DH * 3
    while True:
        raw = proc.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        buf_frames.append(np.frombuffer(raw, np.uint8).reshape(DH, DW, 3))
        if len(buf_frames) == CHUNK:
            q.put((idx, buf_frames))
            idx += len(buf_frames)
            buf_frames = []
    if buf_frames:
        q.put((idx, buf_frames))
    q.put(None)
    proc.wait()


class GpuWorker:
    def __init__(self, device):
        self.dev = device
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip = self.clip.to(device).eval()
        self.clip_mean = torch.tensor([0.4815, 0.4578, 0.4082], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.2686, 0.2613, 0.2758], device=device).view(1, 3, 1, 1)
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        gy, gx = torch.meshgrid(torch.arange(FH, device=device, dtype=torch.float32),
                                torch.arange(FW, device=device, dtype=torch.float32),
                                indexing='ij')
        self.base_grid = torch.stack([gx, gy], -1)  # H,W,2

    @torch.no_grad()
    def clip_feats(self, rgb_gpu):  # B,3,H,W float 0-1
        x = torch.nn.functional.interpolate(rgb_gpu, (224, 224), mode='bilinear',
                                            align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        with torch.autocast('cuda'):
            f = self.clip.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def raft_warp_residual(self, small_gpu):  # B,3,FH,FW float 0-1 (连续子采样帧)
        a = small_gpu[:-1] * 2 - 1
        b = small_gpu[1:] * 2 - 1
        with torch.autocast('cuda'):
            flow = self.raft(a, b)[-1].float()  # B-1,2,H,W: t -> t+1
        # 用 flow 把 t+1 反向 warp 回 t: 采样坐标 = base + flow
        g = self.base_grid.unsqueeze(0) + flow.permute(0, 2, 3, 1)
        g[..., 0] = g[..., 0] / (FW - 1) * 2 - 1
        g[..., 1] = g[..., 1] / (FH - 1) * 2 - 1
        warped = torch.nn.functional.grid_sample(small_gpu[1:], g, mode='bilinear',
                                                 padding_mode='border', align_corners=True)
        return (warped - small_gpu[:-1]).abs().mean(dim=(1, 2, 3)) * 255.0


def scan(path, device='cuda:0'):
    fps = probe_meta(path)
    torch.cuda.set_device(device)
    t_load0 = time.time()
    worker = GpuWorker(device)
    load_s = time.time() - t_load0

    q = Queue(maxsize=4)
    th = threading.Thread(target=decoder, args=(path, q), daemon=True)
    t0 = time.time()
    th.start()

    lum, d1_parts, hsv_diff = [], [], []
    feats, warps = [], []
    prev_rgb_gpu = None           # 上一 chunk 末帧(GPU, 1,3,H,W)
    carry_small = None            # 跨 chunk 的最后一帧(RAFT 连续性)
    n = 0

    def rgb_to_hsv_gpu(x):        # x: B,3,H,W 0-1 -> 近似 OpenCV HSV 量纲(H:0-180,S/V:0-255)
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        mx, _ = x.max(1); mn, _ = x.min(1)
        df = mx - mn + 1e-8
        h = torch.zeros_like(mx)
        m = (mx == r); h[m] = (60 * (g - b) / df)[m] % 360
        m = (mx == g); h[m] = (60 * (b - r) / df + 120)[m]
        m = (mx == b); h[m] = (60 * (r - g) / df + 240)[m]
        return torch.stack([h / 2, df / (mx + 1e-8) * 255, mx * 255], 1)

    while True:
        item = q.get()
        if item is None:
            break
        idx, frames = item
        arr = np.stack(frames)
        # ---- 整个 chunk 一次上 GPU: B,3,DH,DW RGB 0-1 ----
        rgb = torch.from_numpy(arr[..., ::-1].copy()).to(device, non_blocking=True)
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
        gray = rgb.mean(1)                                     # B,H,W (0-1)
        lum.extend((gray.mean((1, 2)) * 255).cpu().tolist())
        g_all = torch.cat([prev_rgb_gpu.mean(1), gray]) if prev_rgb_gpu is not None else gray
        d1_parts.extend((g_all.diff(dim=0).abs().mean((1, 2)) * 255).cpu().tolist())
        hsv = rgb_to_hsv_gpu(torch.cat([prev_rgb_gpu, rgb]) if prev_rgb_gpu is not None else rgb)
        hsv_diff.extend(hsv.diff(dim=0).abs().mean((1, 2, 3)).cpu().tolist())
        prev_rgb_gpu = rgb[-1:]
        # ---- 子采样帧 CLIP + RAFT ----
        sel = [k for k in range(len(frames)) if (idx + k) % SUB_EVERY == 0]
        if sel:
            sub = rgb[sel]
            small = torch.nn.functional.interpolate(sub, (FH, FW), mode='bilinear',
                                                    align_corners=False)
            feats.append(worker.clip_feats(sub))
            if carry_small is not None:
                small = torch.cat([carry_small, small])
            if small.shape[0] >= 2:
                warps.append(worker.raft_warp_residual(small))
            carry_small = small[-1:]
        n = idx + len(frames)
    torch.cuda.synchronize()
    compute_s = time.time() - t0

    feats = torch.cat(feats)
    clip_d = torch.zeros(feats.shape[0])
    clip_d[1:] = 1 - (feats[1:] * feats[:-1]).sum(-1).cpu()
    warp = torch.cat([torch.zeros(1, device=device)] + warps).cpu().numpy()
    la = np.asarray(lum)
    flick = np.zeros(n)
    flick[1:-1] = np.abs(la[1:-1] - (la[:-2] + la[2:]) / 2)
    hs = np.asarray(hsv_diff)
    cuts = [int(i) for i in np.where(hs > 27.0)[0]]   # ContentDetector 同款阈值

    dur = n / fps
    out = {'video': os.path.basename(path), 'n_frames': n, 'fps': fps,
           'duration_s': round(dur, 2), 'device': torch.cuda.get_device_name(device),
           'model_load_s': round(load_s, 2), 'compute_s': round(compute_s, 2),
           'realtime_factor': round(compute_s / dur, 3),
           'signals': {'luminance': np.round(la, 2).tolist(),
                       'diff_d1': np.round(np.asarray(d1_parts), 2).tolist(),
                       'flicker': np.round(flick, 2).tolist(),
                       'clip_dist': np.round(clip_d.numpy(), 4).tolist(),
                       'warp_residual': np.round(warp, 2).tolist()},
           'cuts_frames': cuts}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save', action='store_true')
    a = ap.parse_args()
    r = scan(a.video, a.device)
    print(json.dumps({k: v for k, v in r.items() if k != 'signals'},
                     ensure_ascii=False, indent=1))
    if a.save:
        p = os.path.join(BASE, 'probes', f"fast_{r['video'].replace('.mp4','')}.json")
        json.dump(r, open(p, 'w'))
        print('saved', p)
