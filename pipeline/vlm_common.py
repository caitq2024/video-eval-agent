#!/usr/bin/env python3
"""E0/E3 共用：contact sheet 构建 + Bedrock Nova 调用 + JSON 解析。"""
import json
import re

import boto3
import cv2
import numpy as np

MODEL_ID = "us.amazon.nova-2-lite-v1:0"
_client = None

# 合成视频对应的"生成 prompt"，作为 VLM 评审的条件输入
GEN_PROMPT = ("A single continuous shot: a textured dark background with scattered colored "
              "rectangles, camera slowly panning right at constant speed. An orange smiley-face "
              "ball moves smoothly along a sine path and stays fully visible in frame at all times. "
              "Constant lighting, no scene changes, no cuts.")

DEFECT_SCHEMA = """Respond ONLY with a JSON object, no markdown fence, in this exact schema:
{
  "defect": true|false,
  "findings": [
    {"type": "temporal_jump|flicker|freeze|unexpected_cut|subject_out_of_frame|black_frame|corrupt_frame|other",
     "start_s": <number>, "end_s": <number>, "severity": 1-5,
     "evidence": "<short reason>", "confidence": 0.0-1.0}
  ]
}
If the video looks fine, return {"defect": false, "findings": []}."""


def client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name="us-west-2")
    return _client


def contact_sheet(frames_with_ts, cols, tile_w=240):
    """frames_with_ts: [(frame_bgr, t_seconds)]，铺网格并在每格标注时间戳"""
    tiles = []
    for f, ts in frames_with_ts:
        h, w = f.shape[:2]
        tile = cv2.resize(f, (tile_w, int(tile_w * h / w)))
        cv2.rectangle(tile, (0, 0), (78, 20), (0, 0, 0), -1)
        cv2.putText(tile, f"{ts:.2f}s", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
        tiles.append(tile)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def ask_nova(blocks, temperature=0.0, max_tokens=800):
    """blocks: converse content blocks。返回 (parsed_json_or_None, raw_text)"""
    r = client().converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": blocks}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature})
    text = r["output"]["message"]["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, text
    try:
        return json.loads(m.group(0)), text
    except json.JSONDecodeError:
        return None, text


def img_block(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return {"image": {"format": "png", "source": {"bytes": buf.tobytes()}}}


def video_block(path):
    with open(path, "rb") as f:
        return {"video": {"format": "mp4", "source": {"bytes": f.read()}}}


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps
