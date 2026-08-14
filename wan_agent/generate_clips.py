#!/usr/bin/env python3
"""生成 Agent：把分镜 wan_prompt 派发到空闲 GPU 上跑 Wan generate.py。

多卡并行：一个 GPU 同时只跑一个任务（1.3B ~29G / 5B offload ~34G，均需近整卡）。
每个任务记录 wall time / 返回码 / 日志尾部，失败也保留 —— 失败本身是评估素材(T0)。
"""
import os
import queue
import subprocess
import threading
import time

from common import MODELS, PY

HF_ENV = {'HF_HOME': '/home/ec2-user/hf_cache', 'TORCH_HOME': '/home/ec2-user/torch_cache'}


def free_gpus(min_free_mb=36000):
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total = [int(x) for x in line.split(',')]
        if total - used >= min_free_mb:
            gpus.append(idx)
    return gpus


def run_one(job, gpu):
    """job: {model, prompt, seed, out_path, duration_s}"""
    m = MODELS[job['model']]
    frames = m['frames_for'](job['duration_s'])
    cmd = [PY, 'generate.py', '--task', m['task'], '--size', m['size'],
           '--frame_num', str(frames), '--ckpt_dir', m['ckpt'],
           '--base_seed', str(job['seed']), '--prompt', job['prompt'],
           '--save_file', job['out_path']] + m['extra']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **HF_ENV)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=m['codebase'], env=env,
                       capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    ok = r.returncode == 0 and os.path.exists(job['out_path'])
    return {**{k: v for k, v in job.items() if k != 'prompt'},
            'frames': frames, 'size': m['size'], 'fps': m['fps'], 'gpu': gpu,
            'ok': ok, 'gen_wall_s': round(elapsed, 1),
            'log_tail': '' if ok else r.stderr[-2000:]}


def run_jobs(jobs, gpus=None, log=print):
    """并行执行所有生成任务，返回与 jobs 同序的结果列表。"""
    gpus = gpus or free_gpus()
    if not gpus:
        raise RuntimeError('no free GPU')
    log(f'[gen] {len(jobs)} jobs on GPUs {gpus}')
    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    results = [None] * len(jobs)
    lock = threading.Lock()

    def worker(i, job):
        gpu = gpu_pool.get()
        try:
            log(f'[gen] start job{i} {job["model"]} seed={job["seed"]} gpu{gpu} '
                f'-> {os.path.basename(job["out_path"])}')
            try:
                res = run_one(job, gpu)
            except subprocess.TimeoutExpired:
                res = {**{k: v for k, v in job.items() if k != 'prompt'},
                       'gpu': gpu, 'ok': False, 'gen_wall_s': 3600.0,
                       'log_tail': 'TIMEOUT 3600s'}
            with lock:
                results[i] = res
            log(f'[gen] done  job{i} ok={res["ok"]} {res["gen_wall_s"]}s')
        finally:
            gpu_pool.put(gpu)

    threads = [threading.Thread(target=worker, args=(i, j)) for i, j in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
