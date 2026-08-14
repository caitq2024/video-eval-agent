#!/usr/bin/env python3
"""交互式演示服务器：客户输入创意 + 分镜数（1-3）→ 现场生成 + 质量评估。

- 静态托管 docs/（与 GitHub Pages 同一套前端），并挂载 wan_outputs 供播放
- POST /api/submit {idea, n_shots, models} → job_id（后台线程排队执行）
- GET  /api/status/<job_id> → 阶段进度 + 产物；GET /api/jobs → 历史任务

用法（GPU 机器）:
  HF_HOME=/home/ec2-user/hf_cache TORCH_HOME=/home/ec2-user/torch_cache \
  nohup /opt/pytorch/bin/python3 demo_server.py --port 8008 > demo_server.log 2>&1 &
浏览器访问 http://<gpu-ip>:8008/demo.html （注意安全组放行端口）
"""
import argparse
import json
import os
import queue
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory

from common import EVAL_ROOT, MODELS, OUT_ROOT
from director import direct
from generate_clips import free_gpus, run_jobs
from run_pipeline import stitch_prompt

DOCS = os.path.join(EVAL_ROOT, 'repo', 'docs')
app = Flask(__name__)
JOBS = {}                     # job_id -> state dict（内存态；产物落盘 wan_outputs）
WORK_Q = queue.Queue()        # 串行执行：生成占整卡，评估占一卡，避免任务间打架


def set_state(jid, **kw):
    JOBS[jid].update(kw)
    JOBS[jid]['updated'] = time.time()


def run_job(jid):
    st = JOBS[jid]
    idea, n_shots, models = st['idea'], st['n_shots'], st['models']
    pdir = os.path.join(OUT_ROOT, jid)
    os.makedirs(pdir, exist_ok=True)
    try:
        # 1 导演
        set_state(jid, stage='directing', detail='导演 Agent 拆分镜中…')
        t0 = time.time()
        meta = direct(idea, n_shots)
        meta['id'] = jid
        meta['director_wall_s'] = round(time.time() - t0, 1)
        json.dump(meta, open(os.path.join(pdir, 'shots.json'), 'w'),
                  ensure_ascii=False, indent=1)
        set_state(jid, shots=meta['shots'], title=meta.get('title_zh') or meta.get('title'))
        # 2 生成
        jobs = []
        for mk in models:
            os.makedirs(os.path.join(pdir, mk), exist_ok=True)
            for s in meta['shots']:
                jobs.append({'model': mk, 'prompt': s['wan_prompt'],
                             'seed': 4200 + s['shot_id'],
                             'out_path': os.path.join(pdir, mk, f'shot{s["shot_id"]}.mp4'),
                             'duration_s': s['duration_s'],
                             'pid': jid, 'shot_id': s['shot_id']})
        est = max(5, 9 * ((len(jobs) - 1) // max(1, len(free_gpus())) + 1))
        set_state(jid, stage='generating',
                  detail=f'Wan 生成 {len(jobs)} 段 clip（预计 ~{est} 分钟）…',
                  clips_done=0, clips_total=len(jobs))

        def log(msg):
            if '[gen] done' in msg:
                set_state(jid, clips_done=st.get('clips_done', 0) + 1,
                          detail=f'已生成 {st.get("clips_done", 0)}/{len(jobs)} 段 clip')

        results = run_jobs(jobs, log=log)
        gen = meta.setdefault('generation', {})
        for r in results:
            gen.setdefault(r['model'], {})[f'shot{r["shot_id"]}'] = {
                k: r[k] for k in ('ok', 'gen_wall_s', 'frames', 'size', 'fps',
                                  'gpu', 'seed', 'log_tail') if k in r}
        # 3 拼接
        set_state(jid, stage='stitching', detail='ffmpeg 转场拼接中…')
        meta = stitch_prompt(jid, meta, models)
        json.dump(meta, open(os.path.join(pdir, 'shots.json'), 'w'),
                  ensure_ascii=False, indent=1)
        # 4 评估
        from evaluate import evaluate
        dev = f'cuda:{(free_gpus() or [0])[0]}'
        evals = {}
        for mk in models:
            set_state(jid, stage='evaluating', detail=f'质量评估 Agent 检测 {mk} 成片…')
            evals[mk] = evaluate(jid, mk, dev)
        set_state(jid, stage='done', detail='完成',
                  scores={mk: e['scores']['total'] for mk, e in evals.items()})
    except Exception as e:
        traceback.print_exc()
        set_state(jid, stage='error', detail=f'{type(e).__name__}: {e}'[:300])


def worker():
    while True:
        jid = WORK_Q.get()
        try:
            run_job(jid)
        finally:
            WORK_Q.task_done()


@app.post('/api/submit')
def submit():
    d = request.get_json(force=True)
    idea = (d.get('idea') or '').strip()
    if not idea or len(idea) > 300:
        return jsonify({'error': '请输入 1-300 字的创意描述'}), 400
    n_shots = max(1, min(3, int(d.get('n_shots', 3))))
    models = [m for m in d.get('models', ['wan2.2']) if m in MODELS] or ['wan2.2']
    jid = 'demo_' + uuid.uuid4().hex[:8]
    JOBS[jid] = {'job_id': jid, 'idea': idea, 'n_shots': n_shots, 'models': models,
                 'stage': 'queued', 'detail': f'排队中（前面还有 {WORK_Q.qsize()} 个任务）',
                 'created': time.time()}
    WORK_Q.put(jid)
    return jsonify({'job_id': jid})


@app.get('/api/status/<jid>')
def status(jid):
    if jid not in JOBS:
        # 服务器重启后从磁盘恢复已完成任务
        if os.path.exists(os.path.join(OUT_ROOT, jid, 'shots.json')):
            return jsonify({'job_id': jid, 'stage': 'done', 'from_disk': True})
        return jsonify({'error': 'unknown job'}), 404
    return jsonify(JOBS[jid])


@app.get('/api/jobs')
def jobs():
    return jsonify(sorted(
        [{k: v for k, v in j.items() if k != 'shots'} for j in JOBS.values()],
        key=lambda x: -x['created']))


@app.get('/api/result/<jid>')
def result(jid):
    """demo 任务的完整产物（shots.json + 各模型 eval.json），前端渲染用。"""
    pdir = os.path.join(OUT_ROOT, jid)
    sp = os.path.join(pdir, 'shots.json')
    if not os.path.exists(sp):
        return jsonify({'error': 'not found'}), 404
    meta = json.load(open(sp))
    out = {'meta': meta, 'evals': {}}
    for mk in MODELS:
        ep = os.path.join(pdir, mk, 'eval.json')
        if os.path.exists(ep):
            out['evals'][mk] = json.load(open(ep))
    return jsonify(out)


@app.get('/outputs/<path:p>')
def outputs(p):
    return send_from_directory(OUT_ROOT, p)


@app.get('/')
@app.get('/<path:p>')
def static_docs(p='index.html'):
    return send_from_directory(DOCS, p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8008)
    a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    print(f'demo server on 0.0.0.0:{a.port} — 浏览器打开 http://<本机IP>:{a.port}/demo.html')
    app.run(host='0.0.0.0', port=a.port, threaded=True)
