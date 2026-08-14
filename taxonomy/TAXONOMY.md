# 视频生成 Agent 常见错误分类（v1，evaluation-first）

日期：2026-08-13。依据网上调研（VBench/VBench++、VideoScore/VideoFeedback、EvalCrafter、
T2V-CompBench、VideoPhy/VideoPhy-2、Physics-IQ、VANE-Bench、Sora 综述 arXiv:2402.17177、
AIGVE Survey arXiv:2410.19884）整理，按"检查层级由浅入深"排序。
每类给出：定义、检测信号、拦截层，以及本次合成测试的拦截结果（详见 e6_interception.json）。

## 分类表（10 类核心 + 3 类暂缓）

| # | 错误类型 | 定义与典型表现 | 文献依据 | 检测信号 | 拦截层 |
|---|---|---|---|---|---|
| T0 | 文件/编码层故障 | 0 字节/截断/无法解码、时长与请求不符、帧率异常 | 学术基准不覆盖，工程红线（AIGVE Survey 也指出） | ffprobe gate | detector 直判 |
| T1 | 跳帧/瞬移 (temporal jump) | 丢帧导致内容瞬移、动作一帧一跳 | VBench motion_smoothness；VideoScore"卡顿" | flow-warp residual、CLIP 特征跳变、PySceneDetect | detector + VLM 定向 |
| T2 | 时序闪烁 (flickering) | 单帧亮度/颜色突变，静态区域"呼吸" | VBench temporal_flickering；EvalCrafter Warping Error | 亮度尖峰（全帧率） | detector 直判 |
| T3 | 冻结/伪静态 (freeze) | 画面突然静止；或全片近静态"换稳定性" | VBench dynamic_degree（专门惩罚伪静态）；EvalCrafter Flow-Score（Gen2/Pika 实测 0.5-0.7 极低） | 帧差≈0 连续段；全片运动量统计 | detector 直判 |
| T4 | 意外场景切换 (unexpected cut) | 单镜头任务中出现硬切/背景突换 | VANE-Bench sudden appearance；VideoScore"镜头突兀切换" | PySceneDetect + 任务约束（单镜头?） | detector + 任务 rubric |
| T5 | 主体出界/裁切 (framing) | 主体被边缘裁切或无理由离开画面 | 学术覆盖薄弱（散见 Sora 综述），客户高频痛点 | 主体检测/跟踪 + border touch + 可见面积骤降 | 候选→VLM 按 prompt 裁决（必须区分特写/有意出画） |
| T6 | 黑帧/坏帧 (black/corrupt) | 全黑、纯色、乱码噪声帧 | VideoScore"斑点黑块"；工程实践 | 亮度阈值、噪声统计 | detector 直判 |
| T7 | 主体变形 (deformation) | 肢体/刚体解剖学不可能的形变、六指、橡皮化 | VANE-Bench unnatural transformations；VideoScore"反关节" | bbox/mask 纵横比与形状突变、关键点完整度 | detector 候选 + VLM 确认 |
| T8 | 主体身份漂移 (identity drift) | 人物换脸、衣服变色、狗渐变成猫 | VBench subject_consistency；EvalCrafter Face Consistency | 外观特征轨迹（HSV/DINO）断裂 + VLM 对照 prompt | 候选→VLM 分类裁决 |
| T9 | 物体凭空消失/出现 (object permanence) | 次要物体无因果消失、无关元素乱入 | VANE-Bench disappearance；Sora 综述"乱入" | 多目标检测/跟踪的轨迹中断（当前弱项，见结果） | 检测/分割升级项 |
| T10 | 语义不符 (semantic misalignment) | 内容缺失/错配 prompt 的对象、属性、数量、空间关系 | VBench 9 个语义维度；T2V-CompBench 7 类组合性 | 全局 VLM 对齐评分（稀疏采样对全局语义足够） | 全局 VLM 层 |

暂缓（本轮不合成测试，接入真实数据后处理）：

| 类型 | 原因 | 建议信号 |
|---|---|---|
| 物理违反/因果断裂 | 玩具场景难以有意义地合成；文献显示这是重灾区（VideoPhy：最佳模型仅 39.6% 双达标；VideoPhy-2 困难集 22%） | 轨迹速度/加速度/jerk + VideoPhy 类物理 rubric + VLM |
| 穿模/交互失真 | 需要多物体 3D 交互场景 | 分割 mask 重叠分析 + VLM |
| 画面文字错误 | 需要含文字的生成场景 | OCR（EvalCrafter OCR-Score 做法） |

## 实测拦截率（每类 2 个变体，合成注入，hybrid pipeline，2026-08-13）

| 类型 | 拦截 | 拦截路径 | 说明 |
|---|---|---|---|
| T1 跳帧/瞬移 | **2/2** | PySceneDetect（大跳）+ warp 残差绝对阈值直判（小跳） | 小跳（8帧）靠新增的 warp>4.5 探测器规则，正常视频 warp max ≤3.1，零误报 |
| T2 闪烁 | **2/2**（5/5 事件） | 亮度尖峰 detector 直判 | 全帧率扫描，单帧也不漏 |
| T3 冻结 | **2/2** | 帧差≈0 连续段 detector 直判 | |
| T4 意外切换 | **2/2** | PySceneDetect + 单镜头任务约束 | |
| T5 主体出界 | **2/2** | 跟踪候选 + VLM 按 prompt 裁决 | 正常特写/正常出画 0 误报（E4） |
| T6 黑帧/坏帧 | **2/2**（3/3 事件） | 亮度阈值 detector 直判 | |
| T7 主体变形 | **2/2** | bbox 纵横比突变 detector 直判 | 压扁/拉长两个方向都拦截 |
| T8 身份漂移 | **2/2**（已升级） | 色相直方图漂移直判（0.95 vs 正常 ≤0.03） | R4 实测：swap 两变体精确命中、零误报；DINOv2 特征对颜色不敏感（0.07），直方图才是对的信号；Claude 裁决亦可修复（E7 5/5） |
| T9 物体消失 | **2/2 定位 / 有误报** | GroundingDINO 实例关联（R4） | 两变体均正确定位（5.06s/3.19s），但逐帧检测抖动在 6 条非目标视频产生假事件——产线需 SAM2 mask 传播级跟踪 |
| T10 语义不符 | **2/2** | 全局 VLM 对齐评分（8 帧均匀） | 两个错误 prompt 都判 alignment=1；正确 prompt 对照=3 不误报 |

初版事件级合计 19/22（86%）；R4 升级后（色相直方图直判修 T8、Claude+rubric 裁决 E7 实测 5/5）
达 **21/22（95%）**，clean 与 3 条正常对照仍零误报。剩余缺口：T9 需 SAM2 级跟踪
（GroundingDINO 实例关联已能定位两个 vanish 事件，但逐帧检测抖动带来误报，未计入拦截）。

## 关键调研结论（影响方案设计）

1. **VANE-Bench 实测：9 个 Video-LMM 大多无法可靠检测细微时空异常**——单靠"把视频丢给
   多模态大模型"不可行，必须探针先定位、VLM 只做裁决。这与我们 E3 的实验发现完全一致。
2. **Physics-IQ：视觉真实感与物理理解不相关**——画面越逼真不代表物理越对，物理层需要
   独立检查，不能用画质分数代替。
3. **EvalCrafter：闭源模型用"近静态"换稳定性**（Flow-Score 0.5-0.7）——时序一致性必须
   与动态程度一起看，否则会奖励"不动的视频"（对应计划 §3 注意事项）。
4. **文件/编码层没人管但工程上最先炸**——ffprobe gate 应是所有视频的第一道检查。
