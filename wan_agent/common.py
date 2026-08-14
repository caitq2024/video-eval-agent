#!/usr/bin/env python3
"""wan_agent 公共设施：路径、Bedrock 调用、ffmpeg。"""
import json
import os
import re
import subprocess

import boto3

WAN_ROOT = '/home/ec2-user/efs/wan'
EVAL_ROOT = '/home/ec2-user/efs/agent_evaluation/video_eval'
FFMPEG = os.path.join(EVAL_ROOT, 'bin', 'ffmpeg')
OUT_ROOT = os.path.join(EVAL_ROOT, 'wan_outputs')
PY = '/opt/pytorch/bin/python3'

CLAUDE = 'global.anthropic.claude-opus-4-8'   # 裁决/导演主模型（客户指定 Opus 4-8）

MODELS = {
    'wan2.1': {
        'codebase': os.path.join(WAN_ROOT, 'Wan2.1'),
        'ckpt': os.path.join(WAN_ROOT, 'Wan2.1-T2V-1.3B'),
        'task': 't2v-1.3B', 'size': '832*480', 'fps': 16,
        'frames_for': lambda dur: int(round(dur * 16 / 4)) * 4 + 1,  # 4n+1
        'extra': ['--offload_model', 'False',
                  '--sample_shift', '8', '--sample_guide_scale', '6'],
    },
    'wan2.2': {
        'codebase': os.path.join(WAN_ROOT, 'Wan2.2'),
        'ckpt': os.path.join(WAN_ROOT, 'Wan2.2-TI2V-5B'),
        'task': 'ti2v-5B', 'size': '1280*704', 'fps': 24,
        'frames_for': lambda dur: int(round(dur * 24 / 4)) * 4 + 1,
        # A100 40GB 放不下 5B 全量(峰值~45G)，offload 后 ~34G；
        # 不加 --t5_cpu：offload 模式下 T5 用完即回 CPU，GPU 编码快 10 min
        'extra': ['--offload_model', 'True', '--convert_model_dtype'],
    },
}

_client = None


def bedrock():
    global _client
    if _client is None:
        _client = boto3.client('bedrock-runtime', region_name='us-west-2')
    return _client


def ask_claude(blocks, max_tokens=1500, model=CLAUDE, retries=4):
    """blocks: converse content blocks。返回 (parsed_json_or_None, raw_text)。
    并行裁决下可能遇到 Bedrock 限流，指数退避重试。"""
    import random
    import time
    for attempt in range(retries + 1):
        try:
            r = bedrock().converse(
                modelId=model,
                messages=[{'role': 'user', 'content': blocks}],
                inferenceConfig={'maxTokens': max_tokens})  # opus-4-8 已弃用 temperature
            break
        except Exception as e:
            if attempt >= retries or 'Throttl' not in type(e).__name__ + str(e):
                raise
            time.sleep(1.5 ** attempt + random.random())
    text = r['output']['message']['content'][0]['text']
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None, text
    try:
        return json.loads(m.group(0)), text
    except json.JSONDecodeError:
        return None, text


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_meta(path):
    """T0 文件层 gate：能否解码、时长、帧数、分辨率。"""
    out = run([FFMPEG, '-i', path]).stderr
    ok = 'Video:' in out
    dur = None
    m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', out)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r'(\d+(?:\.\d+)?) fps', out)
    fps = float(m.group(1)) if m else None
    m = re.search(r'(\d{3,4})x(\d{3,4})', out)
    wh = (int(m.group(1)), int(m.group(2))) if m else None
    return {'decodable': ok, 'duration_s': dur, 'fps': fps, 'resolution': wh}
