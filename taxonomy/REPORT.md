# 视频生成质量评估 PoC 实验报告

日期：2026-08-13 · 环境：CPU-only (无 GPU)，Bedrock `us.amazon.nova-2-lite-v1:0` 做 VLM judge
对应计划：`video_eval/video_eval.md` 的 E0 / E2 / E3 / E4（E1 开源 evaluator、E5-E7 见"下一步"）

## TL;DR

在 9 条带精确 ground truth 的合成缺陷视频上（跳帧 / 单帧闪烁 / 冻结 / 意外硬切 /
主体出界 / 黑帧+坏帧，外加 clean / 正常出画 / 正常特写三条对照）：

| 方案 | 缺陷事件召回 | 误报 | mean temporal IoU |
|---|---|---|---|
| VLM 均匀 8 帧（两个 offset） | 11–22% | 0–2 | ≤0.06 |
| VLM 均匀 32 帧（两个 offset） | **11–33%（随 offset 波动 3 倍）** | 0–1 | ≤0.11 |
| VLM 原生视频输入（Nova 内部抽帧） | **0%** | 0 | 0 |
| VLM 均匀 16 帧（E3 对照） | 33% | 1 | 0.14 |
| REACT 式两阶段（粗看→密查，2 次调用） | 22% | 0 | 0.03 |
| CaC 式窗口路由 + 泛泛提问 | 33% | 0 | 0.03 |
| **全帧探针 + hybrid 裁决（最终方案）** | **100% (9/9)** | **0** | **0.47** |

结论直接验证了计划的核心论点，并补充了一个新发现：

1. **均匀抽帧对稀疏坏帧的召回低且不稳定**——同样 32 帧，offset 挪半步召回从 33% 掉到
   11%；单帧闪烁 12 次尝试无一命中；把 mp4 直接丢给 VLM 更是全漏（内部 ~1fps 抽帧）。
2. **全帧低成本扫描是必须的**：探针层（RAFT warp 残差 + CLIP 特征跳变 + 亮度/帧差统计 +
   PySceneDetect + 主体跟踪）候选窗覆盖 9/9 缺陷，clean 视频 0 误报窗，
   单条 8s 视频 CPU 全帧扫描约 23s（GPU 上预计 <2s）。
3. **新发现：光把"对的窗口"喂给 VLM 还不够**。CaC 式路由窗口选得全对，但对窗口
   泛泛地问"有没有缺陷"，Nova 在跳帧/冻结/出界窗口上全部漏判（召回仍是 33%）。
   瓶颈从"采样"转移到"窗口内判定"。
4. **修复方式是分工（hybrid）**：像素级证据充分的缺陷（黑帧/冻结/硬切/闪烁）由
   detector 直接出结论——时间定位还更准；VLM 只做两件事：
   (a) 语义裁决（出画/贴边是有意构图还是异常裁切，结合 intended prompt）；
   (b) 针对触发信号的**定向提问**（"逐格检查主体位置是否连续"而不是"有没有问题"）。
   这样召回 9/9、0 误报。

## E4 专项：边缘碰撞 ≠ 出界失败

| 方法 | 异常裁切召回 | 正常样本误报（特写/正常出画/clean） |
|---|---|---|
| 纯规则（border touch / 面积骤降） | 1/1 | **2/3**（正常出画、特写都被误杀） |
| 纯 VLM 均匀 16 帧 | **0/1（漏检）** | 1/3 |
| **工具候选 + VLM 按 prompt 裁决** | **1/1** | **0/3** |

与计划 4.2 节的预测完全一致："碰到边缘只能作为候选信号"，最终裁决必须带着
intended prompt 让 VLM 判断是有意构图还是生成缺陷。

## 产物清单

```
experiments/
├── videos/            9 条测试视频 + ground_truth.json（缺陷类型/时间窗/severity）
│                      （scripts/gen_videos.py、gen_e4_videos.py 可复现）
├── probes/            每条视频的全帧探针信号 + anomaly score + 候选窗（JSON）
├── results/
│   ├── e0_sampling.json         E0 逐条记录（含 Nova findings 原文）
│   ├── e3_routing.json          E3 三种路由逐条记录
│   ├── e3b_hybrid.json          E3b hybrid 逐条记录
│   ├── e4_out_of_frame.json     E4 三方法对比
│   ├── summary_rescored.json    统一口径的汇总数字
│   ├── anomaly_timelines.png    信号/融合分数/GT/候选窗可视化（客户演示用）
│   └── REPORT.md                本报告
└── scripts/
    ├── gen_videos.py            合成缺陷视频（E7 的可控故障注入）
    ├── e2_probes.py             全帧探针：RAFT/CLIP/像素统计/PySceneDetect/主体跟踪
    ├── e2_fuse.py               信号融合（robust z-score × 绝对下限门控）+ Top-K 窗口
    ├── e0_vlm_sampling.py       E0 抽帧敏感性
    ├── e3_routing.py            E3 uniform/REACT/CaC 路由对比
    ├── e3b_hybrid.py            E3b hybrid（最终方案）
    ├── e4_out_of_frame.py       E4 出界专项
    └── viz_timeline.py          时间轴可视化
```

## 实现要点（可直接迁移到客户真实视频）

- **融合公式**：每路软信号 = robust z-score（per-video median/MAD）×
  **绝对下限门控**（来自 clean 校准视频 ×1.5 安全系数）。纯 z-score 会在正常视频上
  放大噪声（首轮 clean 出了 15.9 的假峰）；加绝对门控后 clean 为全 0。
  这对应计划"权重先用规则初始化，再用人工标注校准"。
- **主体跟踪**：合成视频用 HSV 色域 + 最大连通域即可；真实视频换
  GroundingDINO/SAM2 zero-shot 检测 + 跟踪，指标（border_touch_rate /
  visible_area_drop / safe_area）不变。
- **RAFT small 在 CPU 上完全可用**：320×184 下 ~0.15s/帧对，无需 GPU 也能全帧扫描。
- **跳帧 vs 硬切的分类学**：12 帧跳变会被 PySceneDetect 报成 cut——时间定位正确，
  类型归并即可，报告里两者都算"时序不连续"一类。
- **一个缺陷多信号共振是特性不是 bug**（如 WRONG SCENE 段同时触发 cut/freeze/主体缺失），
  生产版需要一个按时间重叠合并 findings 的 merge 阶段。

## 局限与下一步

- 合成缺陷 ≠ 真实生成缺陷（纹理/运动分布不同），本轮只验证了**架构与路由**的有效性；
  阈值、权重、rubric 必须用客户真实视频 + 人工标注重新校准（计划第 7 节）。
- E1（VideoScore2 / VideoReward / DOVER / VBench 子项）尚未跑：需要 GPU 与模型下载，
  建议在有 GPU 的环境作为整体质量 judge 基线补充，与本方案互补而非替代。
- VLM 用的是 Nova Lite（客户 demo 环境现成额度）；换 Claude/更强 VLM 预计窗口判定
  召回更高，hybrid 架构不变。
- 下一步按计划推进：客户 300–500 条真实视频 + 双人标注 → 权重校准（逻辑回归/GBDT）
  → E5 sample-specific rubric → E6 分数融合 hard cap → 接入 Agent trace。
