# video-eval-agent — 视频生成 Agent 质量评估

用「全帧探针扫描 → 异常定位 → Hybrid 裁决（detector 直判 + VLM 语义裁决）」评估 AI 生成视频质量。
在 10 类常见缺陷的黄金集上实测拦截 **21/22 事件（95%）**、正常对照零误报；
30s@1080p 单卡 A100 扫描 2.28s（0.076× 实时）。

**闭环验证（本轮新增）**：搭建真实视频生成 Agent（Claude 导演分镜 → Wan2.1-1.3B /
Wan2.2-TI2V-5B 各生成 3×5s → ffmpeg 转场拼接），5 创意 × 2 模型 = 10 成片 30 clip 全部生成成功，
评估管线均值 **~60s/成片**（含 5-13 次 Claude 裁决调用），抓到真实缺陷如
「击剑运动员手中无剑」「静态镜头伪静态」「主体形变」，计划转场点零误报。

**在线展示**：<https://caitq2024.github.io/video-eval-agent/> —— 每条成片的分镜 JSON、
各段 clip、信号时间轴（缺陷窗高亮、点击跳转）、findings 列表、评估耗时分解、双模型对比。

- `wan_agent/` — 视频生成 Agent + 闭环评估编排
  - `director.py` 导演 Agent（Bedrock Claude Sonnet 4.5，idea → 1-3 分镜 JSON，
    输出同时是评估侧的 intended prompt / 转场豁免 / 主体跟踪词 / 运动先验）
  - `generate_clips.py` 多卡并行派发 Wan generate.py；`stitch.py` xfade/concat 拼接
  - `evaluate.py` 评估 Agent：T0 gate → fast_scan → 直判（转场豁免）→ GroundingDINO
    主体探针 → 融合候选窗 + Claude+rubric 裁决 → T10 语义对齐，全阶段计时
  - `run_pipeline.py` / `run_all_evals.py` / `build_site.py` 编排与站点构建
- `pipeline/` — 评估管线：`fast_scan.py`（GPU 全帧扫描）、`e2_fuse.py`（门控融合定位）、
  `e3b_hybrid.py`/`e7_claude_verdict.py`（直判 + Claude/rubric 裁决）、`t8_t9_probes.py`（身份漂移/物体消失探测）
- `taxonomy/` — 10 类缺陷定义（含文献依据与拦截率）、全部实验报告、开源 judge 基线对比
  （VideoScore2 / VideoReward / DOVER / VBench——为什么全局分数抓不住稀疏坏帧）
- `docs/` — GitHub Pages 展示站（纯静态，data.json 驱动）

## 核心结论

1. 均匀抽帧 VLM 对稀疏坏帧召回 11-33% 且随 offset 波动；全帧探针 + Hybrid 裁决同预算下 100%
2. SOTA 全局 judge 同样失灵（VideoScore2 21% / VideoAlign 47% / DOVER 79% 低于 clean）——
   根因是「稀疏采样 + 全局池化」；VBench 特征与探针同源，差在全片均值 vs 逐帧离群
3. 裁决层：像素证据直判不花钱；语义问题给 VLM 时要带证据、rubric 先行（Nova 1/5 → Claude+rubric 5/5）
4. **真实生成视频 ≠ 合成校准集**：Wan 视频运动/纹理底噪大（warp 中位数 5.3 vs 合成 ≤3.1），
   T1 需改「孤立尖峰」判定；T7/T8 像素证据在活体/小目标上不充分，须回归「候选 → VLM 确认」;
   导演分镜元数据（转场点/motion 先验/expected_subjects）反哺评估是闭环的关键设计
