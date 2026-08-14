# video-eval-agent — 视频生成 Agent 质量评估

用「全帧探针扫描 → 异常定位 → Hybrid 裁决（detector 直判 + VLM 语义裁决）」评估 AI 生成视频质量。
在 10 类常见缺陷的黄金集上实测拦截 **21/22 事件（95%）**、正常对照零误报；
30s@1080p 单卡 A100 扫描 2.28s（0.076× 实时）。

- `pipeline/` — 评估管线：`fast_scan.py`（GPU 全帧扫描）、`e2_fuse.py`（门控融合定位）、
  `e3b_hybrid.py`/`e7_claude_verdict.py`（直判 + Claude/rubric 裁决）、`t8_t9_probes.py`（身份漂移/物体消失探测）
- `taxonomy/` — 10 类缺陷定义（含文献依据与拦截率）、全部实验报告、开源 judge 基线对比
  （VideoScore2 / VideoReward / DOVER / VBench——为什么全局分数抓不住稀疏坏帧）
- `wan_agent/` — 视频生成 Agent（导演分镜 → Wan2.1/2.2 生成 → 转场拼接）+ 闭环评估（进行中）
- `docs/` — GitHub Pages 展示站（<https://caitq2024.github.io/video-eval-agent/>）

## 核心结论

1. 均匀抽帧 VLM 对稀疏坏帧召回 11-33% 且随 offset 波动；全帧探针 + Hybrid 裁决同预算下 100%
2. SOTA 全局 judge 同样失灵（VideoScore2 21% / VideoAlign 47% / DOVER 79% 低于 clean）——
   根因是「稀疏采样 + 全局池化」；VBench 特征与探针同源，差在全片均值 vs 逐帧离群
3. 裁决层：像素证据直判不花钱；语义问题给 VLM 时要带证据、rubric 先行（Nova 1/5 → Claude+rubric 5/5）
