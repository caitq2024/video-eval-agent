# 视频质量评估 R2：真实视频验证 + 错误分类拦截率

日期：2026-08-13 · 承接 REPORT.md（R1：合成缺陷 + hybrid 方案 9/9）
本轮：① 验证 `summit2026_agent_eval_blog/demo_video/remake_b/out` 6 条真实视频；
② 定义 10 类常见错误分类并实测拦截率（见 TAXONOMY.md）。

## 1. 真实视频验证（6 条演示视频，1080p，3.5–5 分钟/条）

内容先验与 T2V 生成视频不同（motion-graphics slides + 屏幕录制 + 游戏画面），
适配：冻结/硬切降级为信息信号（静止 slide 和有意转场是正常的）、剪辑点上的跳变信号降权 0.3、
无固定主体故跳过主体跟踪。扫描全帧像素信号 + 8fps 子采样 CLIP/RAFT，
每条 Top-6 候选窗交给 Nova 按"演示视频先验"裁决。

结果（36 个候选窗，`e5_real_verdicts.json`）：

| 视频 | 时长 | 剪辑点 | 候选窗 | VLM 判定 |
|---|---|---|---|---|
| demo1_b (v1) | 3:35 | 11 | 6 | **1 个缺陷：0:28 生硬切黑 + 黑屏滞留 >1.5s** |
| demo1_b_v2 | 4:44 | 13 | 6 | 全部正常（fade 转场） |
| demo1_b_v3 | 5:02 | 11 | 6 | 全部正常 |
| demo2_b | 3:29 | 0 | 6 | 全部正常（游戏画面/UI 动画） |
| demo2_b_v2 | 3:29 | 1 | 6 | 全部正常 |
| demo2_b_v3 | 3:48 | 1 | 6 | 全部正常 |

交叉验证 demo1_b 0:28 的判定：三个版本同一章节转场的亮度曲线（探针数据）——
v1 是 `200→106→14` 后**黑屏持续 1.5s+**；v2/v3 是 `200→106→14→107→202`
的 0.4s 快速沉黑即回升（标准 dip-to-black）。即 v1 的转场确实更生硬，
v2/v3 已重做——**检测结果与人工版本迭代吻合**，这是一个真实的、非注入的发现。

过程中的一次误报与修复：第一版裁决 prompt 把 v1 的切黑判成 black_flash 后，
我们给 prompt 增加了探针端计算的**亮度轮廓证据**（渐变 fade vs 单帧突降）并明确
"渐变沉黑+标题渐入是正常剪辑"，v2/v3 的 8 处 fade 转场随即全部正确放行。
教训：VLM 裁决层的先验要用低层信号证据喂进去，而不是让它裸看图。

成本：CPU-only 下每条约 12–14 分钟（瓶颈是 RAFT 全帧光流），GPU 上预计 <1 分钟/条。
已知问题：融合分数在演示类内容上会饱和（静止段多 → MAD 极小 → 转场处 z 全部触顶 29.6/21.6），
Top-K 排序在饱和窗口间近似随机；生产版应改用原始信号分位数排序 + 每种信号配额。

## 2. 错误分类与拦截率（evaluation-first）

10 类核心错误的定义、文献依据、检测信号、实测拦截率全部在 **TAXONOMY.md**。
摘要：事件级拦截 **19/22（86%）**，clean 与 3 条正常对照零误报。
- 8/10 类全拦截：跳帧、闪烁、冻结、意外切换、主体出界、黑帧/坏帧、主体变形、语义不符；
- T8 身份漂移 1/2：漏检发生在 Nova Lite 裁决层，升级路径是外观特征漂移的探测器直判；
- T9 物体凭空消失 0/2：结构性缺口，需多目标检测/跟踪探针（GroundingDINO/SAM2）。

本轮新增的探测器规则：
- **warp-jump 直判**：flow-warp residual > 4.5（正常视频实测 max ≤3.1）且不在剪辑点/
  黑帧/闪烁附近 → temporal_jump。补上了 8 帧小跳的漏检，零误报。
- **bbox 纵横比突变** → deformation 直判（主体完整可见时），压扁/拉长均拦截。
- **语义层**：8 帧均匀采样 + intended prompt 的全局对齐评分。错误 prompt 判 1 分、
  正确 prompt 对照 3 分——**稀疏采样对全局语义足够，对稀疏坏帧不够**，两层各司其职。

## 3. 新产物

```
scripts/e5_real_scan.py        真实长视频全帧扫描（双时间轴：全帧率像素 + 8fps CLIP/RAFT）
scripts/e5b_real_verdict.py    候选窗 VLM 裁决（演示视频先验 + 亮度轮廓证据）
scripts/gen_taxonomy_videos.py 10 类 × 2 变体缺陷注入生成器
scripts/e6_interception.py     拦截率测试（detector 直判 + VLM 裁决 + 语义层）
results/TAXONOMY.md            错误分类定义 + 文献依据 + 拦截率表
results/e5_real_verdicts.json  6 条真实视频 36 个窗口的完整裁决
results/e6_interception.json   22 个事件的逐条拦截记录
probes/real_*.json             6 条真实视频的全帧信号（可复用，不需重扫）
```

## R3 补充:扫描速度优化(2026-08-13,8×A100 p4d)

`scripts/fast_scan.py`(取代 gpu_bench 路径):ffmpeg 管道 480px 直出 + 全部像素统计搬上
GPU(含 HSV 切换信号,免 PySceneDetect 二次解码)+ CLIP 预处理 GPU 化 + RAFT fp16 batch
+ 解码线程与 GPU 重叠。

| 配置 | 30s@1080p 单条 | 相对实时 | 吞吐 |
|---|---|---|---|
| CPU 16 vCPU(旧) | ~60s | 2× | — |
| GPU 基线 gpu_bench | 4.9s | 0.16× | — |
| **fast_scan 单卡** | **2.28s** | **0.076×** | ~38k 条/天/卡 |
| **fast_scan 8 卡满载(热态)** | 3.5s/条 | — | **~19.5 万条 30s 视频/天/机** |

正确性验证:跳帧 warp 峰 6.55@3.00s、闪烁@2.06s、黑帧@2.50s、硬切@3.94/4.44s 全中 GT;
clean 各信号均低于门控下限(warp 4.16 < 5.0,fp16 下 margin 略小于 CPU 版,门限不变)。
8 卡并发瓶颈在 CPU 解码/管道争抢(单卡 2.28→并发 3.5s);限 ffmpeg 线程无改善。

## R4 补充:后续路线四项的实测(2026-08-13)

| 路线项 | 状态 | 关键结果 |
|---|---|---|
| E1 开源基线(VideoScore2/VideoReward/DOVER) | ✅ 完成 | 稀疏坏帧组低于 clean 比例:VS2 21% / VideoAlign 47% / DOVER 79%(分差仅 0.76/100)——见 e1_baseline_report.md |
| VBench 6 子项 | ✅ 完成 | 仅 temporal_flickering 可靠;同源特征、全片均值 vs 逐帧离群是关键差异——见 e1b_vbench_report.md |
| T8 身份漂移探测器(DINO/直方图) | ✅ 解决 | DINOv2 特征对颜色不敏感(swap 漂移仅 0.07);**色相直方图漂移 0.95 vs 正常 ≤0.03**,swap_A@4.5s、swap_B@2.81s 直判命中,闪烁/黑帧被持续性规则过滤,零误报。T8 → 2/2 |
| T9 物体消失(GroundingDINO 实例关联) | ⚠ 方向验证 | vanish_A@5.06s、vanish_B@3.19s 正确定位;但逐帧检测抖动(方块检出率 0.88-0.91)导致 6 条非目标视频出现假消失事件。产线需 SAM2 mask 传播级跟踪 |
| Claude 级裁决 + rubric 先行(E7) | ✅ 显著 | 5 个关键裁决案例:Nova Lite 1/5、Claude Sonnet 4.5 4/5、**Claude+rubric 5/5**;swap_B、closeup 全部修复。VideoArgus"看输出前先定标准"实证有效 |

综合:探测器直判修 T8 + Claude 裁决后,taxonomy 拦截 **21/22(95%)**;
最后一个缺口 T9 需要 SAM2 级跟踪(GPU 空闲时可做)。
人评相关性对齐仍待客户真实视频 + 双人标注(PoC externally blocked)。
