#!/usr/bin/env python3
"""热态吞吐测试:模型载一次,连扫 N 条。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_scan

video = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'videos',
                     'bench_30s_1080p.mp4')
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
r = fast_scan.scan(video, 'cuda:0')          # 首条(含 CUDA 上下文预热后)
t0 = time.time()
for i in range(n):
    fast_scan.scan(video, 'cuda:0')
dt = time.time() - t0
print(f'warm: {n} scans in {dt:.2f}s = {dt / n:.2f}s/video')
