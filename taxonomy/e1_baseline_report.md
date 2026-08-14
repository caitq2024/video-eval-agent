# E1 基线实验:开源全局视频质量模型 vs 稀疏坏帧

- 日期:2026-08-13;机器:8xA100-40GB(单卡推理);数据:`experiments/videos/tax_*.mp4` 共 20 条(19 缺陷 + tax_clean 对照),8s / 640x360 / 16fps,ground truth 见 `taxonomy_ground_truth.json`(tax_semantic_A 不在 GT 文件中,按语义错误缺陷计入 19 条)。
- 待验假设:**全局质量分数对稀疏坏帧(单帧闪烁、2-3 帧黑帧/跳帧/换脸)召回不足**。
- 判定口径:把 clean 的分数当作最理想阈值,统计"缺陷视频分数严格低于 clean"的比例(below-clean)。这是对全局分数最有利的口径——实际使用中不可能把阈值恰好定在 clean 单点上。

## 1. 跑通的模型

| 模型 | 底座 | 输出维度 | 推理耗时/条 | 安装结论 |
|---|---|---|---|---|
| VideoAlign / VideoReward (KwaiVGI) | Qwen2-VL-2B + 回归头 | VQ / MQ / TA(z 归一)+ Overall | ~1.0s | 跑通,见踩坑 |
| DOVER (VQAssessment) | Swin-T(技术)+ ConvNeXt(美学) | aesthetic / technical / overall(0-100) | ~0.23s | 跑通,最顺利 |
| VideoScore2 (TIGER-Lab) | Qwen2.5-VL-7B,先 CoT 后打分 | visual_quality / text_alignment / physical_consistency(1-5)| ~15.5s | 跑通,官方解析有 bug,自行修复 |

三个目标模型全部跑通,无需 pyiqa 替补。

## 2. 分数明细(clean 加粗)

### DOVER(0-100,越高越好)

| 视频 | 缺陷类型 | 缺陷时长(s) | aesthetic | technical | overall |
|---|---|---|---|---|---|
| **tax_clean.mp4** | clean | - | 91.264 | 7.742 | 35.472 |
| tax_semantic_A.mp4 | semantic_error | - | 86.937 | 7.005 | 30.154 |
| tax_black_A.mp4 | black_frame+corrupt_frame | 0.18 | 87.367 | 7.569 | 31.560 |
| tax_crop_A.mp4 | subject_crop | 1.2 | 89.080 | 7.663 | 33.146 |
| tax_swap_A.mp4 | identity_swap | 0.12 | 89.952 | 7.437 | 33.512 |
| tax_freeze_A.mp4 | freeze | 0.75 | 90.080 | 7.660 | 34.072 |
| tax_swap_B.mp4 | identity_swap | 0.13 | 89.937 | 7.770 | 34.146 |
| tax_flicker_A.mp4 | flicker | 0.18 | 90.645 | 7.499 | 34.326 |
| tax_vanish_B.mp4 | object_vanish | 0.12 | 90.817 | 7.486 | 34.480 |
| tax_freeze_B.mp4 | freeze | 1.0 | 90.507 | 7.713 | 34.603 |
| tax_crop_B.mp4 | subject_crop | 0.8 | 90.458 | 7.796 | 34.714 |
| tax_cut_A.mp4 | unexpected_cut | 0.5 | 90.652 | 7.783 | 34.889 |
| tax_flicker_B.mp4 | flicker | 0.12 | 91.033 | 7.650 | 35.036 |
| tax_jump_B.mp4 | temporal_jump | 0.12 | 90.699 | 7.847 | 35.062 |
| tax_black_B.mp4 | black_frame | 0.19 | 90.969 | 7.727 | 35.116 |
| tax_deform_A.mp4 | deformation | 0.6 | 91.121 | 7.669 | 35.170 |
| tax_deform_B.mp4 | deformation | 0.6 | 91.474 | 7.659 | 35.545 |
| tax_vanish_A.mp4 | object_vanish | 0.5 | 91.678 | 7.821 | 36.100 |
| tax_jump_A.mp4 | temporal_jump | 0.12 | 92.010 | 7.623 | 36.108 |
| tax_cut_B.mp4 | unexpected_cut | 0.38 | 91.797 | 7.846 | 36.290 |

### VideoAlign / VideoReward(z 分数,越高越好)

| 视频 | 缺陷类型 | 缺陷时长(s) | VQ | MQ | TA | Overall |
|---|---|---|---|---|---|---|
| **tax_clean.mp4** | clean | - | 0.269 | -0.080 | -0.837 | -0.647 |
| tax_jump_A.mp4 | temporal_jump | 0.12 | -0.127 | -0.447 | -0.947 | -1.521 |
| tax_crop_A.mp4 | subject_crop | 1.2 | 0.214 | -0.057 | -1.564 | -1.407 |
| tax_cut_B.mp4 | unexpected_cut | 0.38 | 0.186 | -0.266 | -0.890 | -0.970 |
| tax_swap_B.mp4 | identity_swap | 0.13 | 0.367 | -0.012 | -1.100 | -0.745 |
| tax_cut_A.mp4 | unexpected_cut | 0.5 | 0.353 | -0.227 | -0.839 | -0.713 |
| tax_jump_B.mp4 | temporal_jump | 0.12 | 0.228 | -0.085 | -0.845 | -0.703 |
| tax_freeze_B.mp4 | freeze | 1.0 | 0.269 | -0.085 | -0.884 | -0.700 |
| tax_vanish_B.mp4 | object_vanish | 0.12 | 0.311 | 0.051 | -1.038 | -0.676 |
| tax_crop_B.mp4 | subject_crop | 0.8 | 0.297 | -0.063 | -0.892 | -0.657 |
| tax_deform_B.mp4 | deformation | 0.6 | 0.283 | -0.029 | -0.879 | -0.625 |
| tax_flicker_A.mp4 | flicker | 0.18 | 0.256 | 0.022 | -0.889 | -0.611 |
| tax_black_B.mp4 | black_frame | 0.19 | 0.339 | -0.051 | -0.859 | -0.571 |
| tax_semantic_A.mp4 | semantic_error | - | 0.256 | 0.039 | -0.865 | -0.571 |
| tax_vanish_A.mp4 | object_vanish | 0.5 | 0.311 | 0.017 | -0.851 | -0.524 |
| tax_black_A.mp4 | black_frame+corrupt_frame | 0.18 | 0.325 | 0.045 | -0.876 | -0.506 |
| tax_swap_A.mp4 | identity_swap | 0.12 | 0.353 | 0.118 | -0.956 | -0.485 |
| tax_freeze_A.mp4 | freeze | 0.75 | 0.367 | -0.000 | -0.850 | -0.483 |
| tax_flicker_B.mp4 | flicker | 0.12 | 0.297 | 0.067 | -0.837 | -0.473 |
| tax_deform_A.mp4 | deformation | 0.6 | 0.283 | -0.023 | -0.722 | -0.462 |

### VideoScore2(soft 分,1-5/维,total 为三维求和)

| 视频 | 缺陷类型 | 缺陷时长(s) | visual_quality_soft | text_alignment_soft | physical_consistency_soft | total_soft |
|---|---|---|---|---|---|---|
| **tax_clean.mp4** | clean | - | 4.000 | 3.000 | 4.000 | 11.000 |
| tax_semantic_A.mp4 | semantic_error | - | 3.000 | 1.000 | 1.000 | 5.000 |
| tax_cut_A.mp4 | unexpected_cut | 0.5 | 3.000 | 3.000 | 3.000 | 9.000 |
| tax_swap_A.mp4 | identity_swap | 0.12 | 4.000 | 2.000 | 3.000 | 9.000 |
| tax_freeze_A.mp4 | freeze | 0.75 | 4.000 | 3.000 | 3.000 | 10.000 |
| tax_crop_A.mp4 | subject_crop | 1.2 | 4.000 | 4.000 | 3.000 | 11.000 |
| tax_cut_B.mp4 | unexpected_cut | 0.38 | 4.000 | 3.000 | 4.000 | 11.000 |
| tax_jump_A.mp4 | temporal_jump | 0.12 | 4.000 | 3.000 | 4.000 | 11.000 |
| tax_swap_B.mp4 | identity_swap | 0.13 | 4.000 | 3.000 | 4.000 | 11.000 |
| tax_vanish_B.mp4 | object_vanish | 0.12 | 4.000 | 3.000 | 4.000 | 11.000 |
| tax_deform_B.mp4 | deformation | 0.6 | 3.944 | 3.980 | 3.466 | 11.389 |
| tax_black_A.mp4 | black_frame+corrupt_frame | 0.18 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_black_B.mp4 | black_frame | 0.19 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_crop_B.mp4 | subject_crop | 0.8 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_deform_A.mp4 | deformation | 0.6 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_flicker_A.mp4 | flicker | 0.18 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_flicker_B.mp4 | flicker | 0.12 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_freeze_B.mp4 | freeze | 1.0 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_jump_B.mp4 | temporal_jump | 0.12 | 4.000 | 4.000 | 4.000 | 12.000 |
| tax_vanish_A.mp4 | object_vanish | 0.5 | 4.000 | 4.000 | 4.000 | 12.000 |

## 3. 关键量化:below-clean 召回

| 模型 / 维度 | clean 分数 | 缺陷低于 clean | 比例 |
|---|---|---|---|
| dover / overall | 35.472 | 15/19 | 79% |
| dover / aesthetic | 91.264 | 15/19 | 79% |
| dover / technical | 7.742 | 13/19 | 68% |
| videoalign / Overall | -0.647 | 9/19 | 47% |
| videoalign / VQ | 0.269 | 6/19 | 32% |
| videoalign / MQ | -0.080 | 5/19 | 26% |
| videoalign / TA | -0.837 | 17/19 | 89% |
| videoscore2 / total_soft | 11.000 | 4/19 | 21% |
| videoscore2 / visual_quality_soft | 4.000 | 3/19 | 16% |
| videoscore2 / text_alignment_soft | 3.000 | 2/19 | 11% |
| videoscore2 / physical_consistency_soft | 4.000 | 6/19 | 32% |

### 稀疏缺陷(事件总时长 ≤0.31s,≈1-5 帧)vs 长时缺陷(≥0.375s)

- 稀疏组(9 条):black_A, black_B, flicker_A, flicker_B, jump_A, jump_B, swap_A, swap_B, vanish_B
- 长时组(10 条,含 semantic_A 全程语义错误):crop_A, crop_B, cut_A, cut_B, deform_A, deform_B, freeze_A, freeze_B, semantic_A, vanish_A

| 模型 / 总分维度 | 稀疏组 below-clean | 长时组 below-clean |
|---|---|---|
| dover / overall | 8/9 (89%) | 7/10 (70%) |
| videoalign / Overall | 4/9 (44%) | 5/10 (50%) |
| videoscore2 / total_soft | 1/9 (11%) | 3/10 (30%) |

### 分差幅度(以 DOVER overall 为例,0-100 量表)

- 缺陷视频相对 clean 的分差中位数仅 **0.76 分**;
- 要求缺陷比 clean 低 >1 分:只剩 7/19;低 >2 分:只剩 3/19(semantic_A, black_A, crop_A);
- 有 4 条缺陷视频分数反而**高于 clean**(deform_B, vanish_A, jump_A, cut_B)。

## 4. 各类缺陷的敏感性小结

**相对敏感(全局统计上可见/全程存在)**
- `semantic_error`(全程语义错误):三个模型都是各自的最低分或次低分——DOVER overall 30.15(-5.3),VideoScore2 直接给 TA=1、PC=1(total 5 vs clean 11)。全程性的内容错误全局分数抓得住。
- `subject_crop`(主体出画 ~1s):VideoAlign TA 掉 0.73σ(crop_A 是其 TA 最低分),VideoScore2 PC 对 crop_A 给 3。持续约 1 秒、改变画面构图的缺陷有一定信号。
- `unexpected_cut` / 长 `freeze`:VideoAlign MQ 对 cut_A/cut_B 掉 0.15-0.19σ,VideoScore2 对 cut_A/freeze_A 的 PC 给 3。方向对但幅度小。
- `black_frame+corrupt`(black_A,3 帧):只有 DOVER aesthetic 明显掉分(-3.9,因逐帧美学均值被黑帧拉低);但 black_B(单事件 3 帧)只掉 0.36 分,几乎不可分。

**明显不敏感(稀疏坏帧,1-5 帧)**
- `flicker`(单帧闪烁 x2-3):DOVER 掉 0.4-1.1 分(噪声级);VideoAlign Overall 两条都**高于 clean**;VideoScore2 全部满 4/4/4(total 12 > clean 11)。
- `temporal_jump`(2 帧跳帧):VideoScore2 与 VideoAlign 对 jump_B 均给出高于/近似 clean 的分;jump_A 在 VideoAlign 上意外掉分较多(-0.87)但同模型对 jump_B 只 -0.06,不稳定。DOVER 对 jump_A 反而给出**全场最高美学分**。
- `black_frame`(black_B,3 帧):VideoAlign VQ 反而比 clean 高(0.34 vs 0.27);VideoScore2 给 4/4/4。
- `identity_swap`(2 帧笑脸换成方块):VideoScore2 对 swap_B 给 12 > clean;VideoAlign 对 swap_A 反而是高分。
- `object_vanish`(2-8 帧):三个模型均有一条 vanish 高于 clean。

**共性根因**:三个模型都是稀疏时间采样 + 全局池化——DOVER 每视频只采 32 帧的两支视图,VideoAlign 按固定 fps 下采样,VideoScore2 以 2fps(约 16 帧)喂给 VLM。1-3 帧的缺陷大概率根本没被采到;即使采到,也被其余 100+ 正常帧的池化平均稀释。

## 5. 安装踩坑记录

**VideoAlign(KwaiVGI/VideoReward)**
- 官方 environment.yaml 是完整训练环境(deepspeed/wandb 等),推理只需子集:torch 2.3.1 + transformers 4.45.2 + trl 0.8.6 + peft 0.10.0 + accelerate 0.34。
- 官方要求 flash-attn 2.5.8(源码编译很慢):把 `inference.py` 里 `disable_flash_attn2=False` 改为 `True` 走 sdpa,免装 flash-attn,推理结果正常。
- trl 0.8.6 惰性 import 需要 `rich`、`tyro`,requirements 没写,首跑报 `No module named 'rich'`。
- checkpoint 4.8GB(HF `KwaiVGI/VideoReward`),另会自动拉取底座 Qwen2-VL-2B-Instruct;载入 29s,推理 ~1.0s/条。

**DOVER**
- requirements 钉 `torch~=1.13`,直接忽略:装 torch 2.3.1 后 `pip install -e . --no-deps`,完全兼容。
- 权重 236MB 从 GitHub release 直接 wget。20 条视频总耗时 13s(~0.23s/条),三个模型里最快。
- 小坑:`evaluate_a_set_of_videos.py` 会在仓库目录额外写一个 `zero_shot_res_sensehdr.txt`。

**VideoScore2**
- 依赖最干净(README 给了准确 pin:torch 2.6.0 + transformers 4.53.2 + qwen-vl-utils),模型 ~16GB。
- 官方 `vs2_inference.py` 的解析正则 `visual quality:\s*(\d+)` 匹配不上模型实际输出(最终答案是 `(1) visual quality – clarity, smoothness, artifacts: 4`),导致所有分数为 None;自写解析:取 `</think>` 之后的段落按维度名+数字定位,并把字符位置映射回 token 位置计算 soft 分。
- CoT 生成导致耗时 ~15.5s/条(bf16,官方 temperature=0.7 采样);另注意其 think 段里会自我生成 "ground-truth of Dim-x" 字样,属于其 SFT 数据格式残留。
- 环境 locale 为 POSIX,json 落盘需显式 utf-8。

## 6. 结论:与 hybrid 方案的互补关系

1. **假设成立**。对 19 条缺陷视频,即便把阈值放在最理想的 clean 单点:DOVER overall 召回 15/19(但中位分差仅 0.76/100,阈值留裕量到 2 分就跌到 3/19),VideoAlign Overall 只有 9/19,VideoScore2 total 只有 4/19,且三个模型都存在缺陷视频分数**不低于** clean 的倒挂(DOVER 4 条高于;VideoAlign 10 条高于;VideoScore2 10 条高于、5 条持平)。稀疏组(≤5 帧缺陷,9 条)上:VideoAlign 仅 4/9、VideoScore2 仅 1/9 低于 clean;DOVER 零裕量下 8/9,但要求分差 >1 分立即跌到 4/9、>2 分只剩 1/9(black_A),而这些缺陷在人眼里全部是硬伤(severity 3-4)。
2. **全局分数的定位**:适合做"整体质量/美学/文本对齐"的分布级评估与模型选型(它们对全程性的 semantic_error、较长的 crop/cut 有方向正确的信号),不适合做逐条视频的坏帧拦截。
3. **与我们 hybrid 方案(探针 + VLM 定向裁决,19/22 事件拦截)互补**:hybrid 靠逐帧探针(帧差/黑帧/SSIM 等)保证稀疏事件不漏采,VLM 只对探针命中的窗口做定向裁决,恰好补上全局模型"采样稀疏 + 池化稀释"的结构性盲区;反过来,全局模型可以作为 hybrid 之上的分布级质量回归指标(如版本间 DOVER/VideoAlign 均值漂移监控),两者不是竞争关系。
4. 后续如需单一全局分数做粗筛,建议至少改为**分窗打分取 min** 而非全片一分,否则稀疏坏帧在数学上就会被平均掉。
