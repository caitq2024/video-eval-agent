# E1b:VBench 维度分数对稀疏坏帧的区分能力验证

- 日期:2026-08-13
- 环境:8×A100-40GB GPU 机(与其他训练任务共享,每卡仅剩 ~2-5GB 空闲显存)
- 工具:官方 `pip install vbench`(v0.1.5 系),`--mode=custom_input`,conda 独立 env(python 3.10 + torch 2.3.1+cu121)
- 数据:20 条合成缺陷测试视频 `tax_*.mp4`(8s、640×360、16fps、116-128 帧),缺陷为 1-16 帧的稀疏坏帧段;`tax_clean.mp4` 为无缺陷对照
- 原始输出:`e1b_vbench_raw/`;聚合分数:`e1b_vbench_scores.json`

## 一、总览:6 个维度全部跑通

| 维度 | 状态 | 推理耗时(20 条视频) | 目标缺陷是否被排到最低分 |
|---|---|---|---|
| temporal_flickering | 成功 | 14s | **成立**:flicker_A/B 排 1、2 名 |
| motion_smoothness | 成功 | 65s | **不成立**:jump_A/B 仅排 8、9 名,与 clean 差距只有量程的 2-3% |
| subject_consistency | 成功 | 84s(含 DINO 下载) | **不成立**:swap_A/B 排 13、11 名,几乎与 clean 重叠 |
| background_consistency | 成功(需打补丁) | 24s | **成立**(按 cut 为目标):cut_A/B 排 1、2 名 |
| dynamic_degree | 成功 | 125s(含 RAFT 下载) | **无区分**:20 条全部 True(含 freeze 视频) |
| imaging_quality | 成功 | 144s(含 MUSIQ 下载) | **弱**:black 排 4、6 名;deform 反而略高于 clean |

## 二、逐维度分数表(升序 = 越靠前越差)

### 1. temporal_flickering(clean = 0.9869,量程 0.9661-0.9880)

| 排名 | 视频 | 分数 | 缺陷 |
|---|---|---|---|
| 1 | tax_flicker_A | 0.9661 | flicker×3 |
| 2 | tax_flicker_B | 0.9730 | flicker×2 |
| 3 | tax_black_A | 0.9785 | black+corrupt |
| 4 | tax_cut_B | 0.9821 | cut |
| 5 | tax_cut_A | 0.9823 | cut |
| 6 | tax_black_B | 0.9824 | black |
| 7 | tax_semantic_A | 0.9855 | (语义缺陷) |
| 8-9 | tax_jump_A/B | 0.9864/0.9867 | temporal_jump |
| 13 | **tax_clean** | **0.9869** | 无 |
| 19-20 | tax_freeze_A/B | 0.9877/0.9880 | freeze |

**判定:目标缺陷排最低 → 成立**。flicker_A 比 clean 低 0.0209(占全量程 95%),排序完美。混淆:black/cut 也显著低于 clean——本质合理,黑帧和硬切在帧差意义上就是"闪烁";freeze 反而比 clean 还高(静止帧零闪烁),说明该维度分数高≠视频好。共 12/19 条缺陷视频低于 clean。

### 2. motion_smoothness(clean = 0.9941,量程 0.9730-0.9943)

| 排名 | 视频 | 分数 | 缺陷 |
|---|---|---|---|
| 1 | tax_flicker_A | 0.9730 | flicker×3 |
| 2 | tax_flicker_B | 0.9802 | flicker×2 |
| 3 | tax_black_A | 0.9864 | black+corrupt |
| 4-5 | tax_cut_B/A | 0.9877/0.9879 | cut |
| 6 | tax_black_B | 0.9895 | black |
| **8-9** | **tax_jump_A/B(目标)** | **0.9935/0.9937** | temporal_jump |
| 18 | **tax_clean** | **0.9941** | 无 |

**判定:目标缺陷排最低 → 不成立**。2 帧的时间跳变(0.12s)只把 AMT 插帧误差的全片平均拉低 0.0006(量程的 3%),完全被 flicker/black/cut 淹没。该维度实际测的仍是"帧间突变",与 temporal_flickering 排序高度一致(前 6 名完全相同),对"运动连续性断裂"这种目标缺陷不敏感。

### 3. subject_consistency(clean = 0.9930,量程 0.9749-0.9931)

| 排名 | 视频 | 分数 | 缺陷 |
|---|---|---|---|
| 1 | tax_black_A | 0.9749 | black+corrupt |
| 2 | tax_cut_A | 0.9784 | cut |
| 3 | tax_black_B | 0.9809 | black |
| 4 | tax_cut_B | 0.9817 | cut |
| 5 | tax_semantic_A | 0.9862 | (语义缺陷) |
| **11、13** | **tax_swap_B/A(目标)** | **0.9917/0.9922** | identity_swap |
| 18 | **tax_clean** | **0.9930** | 无 |

**判定:目标缺陷排最低 → 不成立**。2 帧的主体替换只让 DINO 相似度均值低于 clean 0.0009-0.0014(量程的 5-8%),排名中游;最低分被 black/cut 拿走(黑帧让 DINO 特征整体崩掉)。全片平均把稀疏的主体不一致稀释殆尽。

### 4. background_consistency(clean = 0.9616,量程 0.9520-0.9704)

| 排名 | 视频 | 分数 | 缺陷 |
|---|---|---|---|
| 1 | tax_cut_A | 0.9520 | cut |
| 2 | tax_cut_B | 0.9547 | cut |
| 3 | tax_swap_B | 0.9555 | identity_swap |
| 4 | tax_black_A | 0.9563 | black+corrupt |
| 11 | **tax_clean** | **0.9616** | 无 |
| 20 | tax_semantic_A | 0.9704 | (语义缺陷) |

**判定:以 unexpected_cut 为该维度目标 → 成立**(cut 排 1、2),CLIP 特征对场景切换敏感,方向正确。但注意:clean 只排第 11,有 9 条缺陷视频分数比 clean 还高;cut_A 与 clean 的差距 0.0097 仅为量程的 52%,整体量程 0.018 极窄。作为逐条筛查阈值不可用。

### 5. dynamic_degree

20 条视频全部输出 True(动态)。freeze_A/B 的 0.75-1s 冻结不足以让 RAFT 全片平均光流跌破阈值。**该维度对本组缺陷完全无区分。**(它设计目标本来是抓"整条视频糊死不动"的退化生成。)

### 6. imaging_quality(MUSIQ,clean = 67.39,量程 65.59-72.22)

最低 5 名:crop_A(65.59)、swap_B(65.72)、swap_A(66.06)、black_B(66.37)、flicker_A(66.55);最高分是 tax_semantic_A(72.22,构图不同的语义缺陷视频)。black 比 clean 低约 1 分(量程的 13-16%),deform 反而比 clean 高。**对稀疏坏帧基本无判别力**——MUSIQ 是逐帧美学/技术质量均值,几帧坏帧摊薄后噪声大于信号。

## 三、安装踩坑记录

1. `pip install vbench` 本身一次通过(conda 新建 env,python 3.10 + torch 2.3.1+cu121)。`detectron2` 等重依赖只有 object-class 类维度才需要,本次 6 个维度不涉及。
2. OpenAI CLIP 不在 PyPI,需手动 `pip install git+https://github.com/openai/CLIP.git`(background_consistency 依赖)。
3. 各维度权重均能自动下载:AMT-S(motion_smoothness)、DINO ViT-B/16(subject_consistency)、RAFT models.zip(dynamic_degree,dropbox 源约 80MB)、MUSIQ-SPAQ(imaging_quality)。机器可联网时无需手工处理。
4. **共享 GPU 的 OOM 坑**:8 块卡都被其他训练占到 36-39GB/40GB。background_consistency 把整条视频 128 帧一次性送进 CLIP encode,在 ~2GB 空闲显存下 OOM。修法:patch 安装包内 `vbench/background_consistency.py`,帧留 CPU、按 4 帧一块搬上 GPU 编码(不改数值结果,只改 batch 方式),加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后通过。其余 5 个维度在残余显存下直接跑通。
5. vbench CLI 经 torch.distributed 启动,子进程失败时外层退出码仍可能是 0,**必须查日志确认**,不能只看返回码。

## 四、结论:VBench 专项维度 vs 我们的探针

1. **同源性确认**:VBench 每个维度确实就是一个专项算法——temporal_flickering ≈ 静区相邻帧 MAE,motion_smoothness ≈ AMT 插帧残差,subject_consistency ≈ DINO 相邻帧+首帧相似度,background_consistency ≈ CLIP 相似度,dynamic_degree ≈ RAFT 光流,imaging_quality ≈ MUSIQ。与我们探针(帧差/DINO/光流)技术上同一家族。
2. **关键差别在聚合方式**:VBench 把逐帧信号做**全片平均**输出一个标量,稀疏坏帧(2 帧/128 帧)被稀释成 2-8% 量程的微小偏移,淹没在视频内容差异的噪声里。这正是 motion_smoothness 抓不到 jump、subject_consistency 抓不到 swap 的原因。我们的探针用的是同样的逐帧信号,但做**时序峰值检测**(找离群帧),所以能定位稀疏坏帧——差别不在特征,在统计量(mean vs max/outlier)。
3. **能用的场景**:flicker 类"高频、多次出现"的缺陷即使被平均也能压低分数(flicker_A 三段缺陷 → 95% 量程分离),cut 这种改变全局特征分布的缺陷也行。即:缺陷时长占比够大或重复次数够多时,VBench 均值分数才有区分度。
4. **判定**:VBench 维度分数**适合模型级/批次级排名**(大量样本平均后系统性差异会显现),**不适合逐条视频的稀疏坏帧筛查**——6 个维度中仅 temporal_flickering 一个能对其目标缺陷做出可靠的逐条排序,且量程窄(0.966-0.988)、无自然阈值、freeze 还会反向得高分。逐条坏帧筛查仍应使用我们的探针(逐帧信号+离群检测)+ VLM 定向裁决的 hybrid 方案;VBench 可作为模型选型阶段的补充参考,不替代坏帧拦截。
