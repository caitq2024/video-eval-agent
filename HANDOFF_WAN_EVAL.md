# 交接文档：Wan 视频生成 Agent + 质量评估平台

写给 GPU 机器（8×A100 内部机器）上的 Claude Code。日期 2026-08-14。
本文档由 CPU 机器上的会话生成，包含全部必要上下文。**先通读再动手。**

## 0. 一句话背景

我们在给客户做「视频生成 Agent 的质量评估」PoC：全帧探针扫描 → 异常窗定位 → Hybrid 裁决
（detector 直判 + VLM 语义裁决）。合成缺陷视频上已验证 10 类缺陷拦截 21/22（95%）、零误报。
**现在要用真实的生成视频闭环验证**：搭一个视频生成 Agent（导演 → 3 分镜 → 3 段生成 → 转场拼接），
用 Wan2.1/Wan2.2 各生成 5 条 5s 视频，用我们的评估管线检测，做成 GitHub Pages 可展示平台。

## 1. 环境速查

| 事项 | 值 |
|---|---|
| 本机（GPU）python | `/opt/pytorch/bin/python3`（torch 2.8+cu129；已装 opencv-headless/open_clip/scenedetect/transformers） |
| conda 环境 | `~/miniconda3/envs/vbench`（VBench 用，别动）、`benchmark` |
| ffmpeg | 无系统安装；用共享静态版 `/home/ec2-user/efs/agent_evaluation/video_eval/bin/ffmpeg` |
| HF/torch 缓存 | `export HF_HOME=/home/ec2-user/hf_cache TORCH_HOME=/home/ec2-user/torch_cache`（放本地盘，别放 EFS） |
| ⚠ GPU 占用 | 8 张卡经常被用户自己的训练占到 ~38GB/40GB。**动手前 `nvidia-smi`**，挑空闲卡 `CUDA_VISIBLE_DEVICES=N`；Wan2.2-A14B 需要整卡，必要时和用户确认哪张卡可用 |
| Bedrock（us-west-2） | VLM 裁决用。可用模型：`us.amazon.nova-2-lite-v1:0`（便宜）、`global.anthropic.claude-sonnet-4-5-20250929-v1:0`（裁决质量高，实测 5/5 vs Nova 1/5）、`us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| GitHub | 仓库 **caitq2024/video-eval-agent**（见 §6） |
| EFS | 两台机器共享 `/home/ec2-user/efs`，所有代码/结果直接互通 |

## 2. 现有资产地图（`/home/ec2-user/efs/agent_evaluation/video_eval/`）

```
experiments/
├── scripts/
│   ├── fast_scan.py        ★ GPU 快速全帧扫描（30s@1080p 单卡 2.28s）。scan(path, device) 返回
│   │                         全帧信号 {luminance, diff_d1, flicker, clip_dist, warp_residual} + cuts_frames
│   ├── e2_fuse.py            信号融合：robust z-score × 绝对下限门控 → anomaly score → Top-K 候选窗
│   ├── e3b_hybrid.py         detector 直判规则（黑帧/冻结/硬切/闪烁）+ 定向 VLM 裁决（参考实现）
│   ├── e7_claude_verdict.py  ★ Claude 裁决 + rubric 先行（照抄这里的 prompt 模板和 ask() 函数）
│   ├── t8_t9_probes.py       GroundingDINO 实例关联 + 色相直方图漂移（T8/T9 探测器）
│   ├── e5_real_scan.py       真实长视频版扫描（演示视频先验：冻结/转场降权）
│   ├── vlm_common.py         Bedrock converse 封装、contact_sheet 拼图、read_frames
│   └── gen_videos.py 等      合成缺陷视频生成器（不用动）
├── videos/                   合成测试视频 + taxonomy_ground_truth.json
├── probes/                   已扫描信号 JSON（复用，勿删）
└── results/
    ├── TAXONOMY.md         ★ 10 类缺陷定义 + 检测信号 + 拦截率（前端要展示的分类体系）
    ├── REPORT.md / REPORT_R2.md  全部实验记录
    ├── e1_baseline_report.md / e1b_vbench_report.md  开源 judge 基线
    └── *.json                各实验原始结果
```

### 检测阈值速查（合成视频上校准，真实视频可能要重校准——见 §5 注意事项）

| 信号 | 直判规则 | 缺陷类型 |
|---|---|---|
| luminance < 12（连续段） | 直判 | T6 黑帧 |
| diff_d1 < 0.05（连续 ≥3 帧） | 直判 | T3 冻结（生成视频先验下；演示类内容要降权） |
| flicker（亮度对前后帧均值偏差）> 1.5，孤立帧 | 直判 | T2 闪烁 |
| warp_residual > 4.5 且不在剪辑点/黑帧附近（正常 ≤3.1，fp16 下 ≤4.2） | 直判 | T1 跳帧 |
| PySceneDetect/HSV 帧差 > 27 且任务要求单镜头 | 直判 | T4 意外切换（**拼接成片的转场点是预期切换，要用分镜元数据豁免**） |
| 主体 bbox 纵横比 log 偏移 > 0.3（完整可见时） | 直判 | T7 变形 |
| 主体 crop 色相直方图对首秒参考余弦距离 > 0.5 且持续 ≥2 子采样帧 | 直判 | T8 身份漂移（正常 ≤0.03；DINOv2 特征对颜色不敏感，别用） |
| 主体缺失/面积骤降 <0.45×中位 | → VLM 裁决 | T5 出界（区分有意出画/特写） |
| 8 帧均匀采样 + intended prompt 对齐评分 ≤2 | 直判 | T10 语义不符 |
| 软信号候选窗（融合分>2.5 的 Top-K） | → VLM 定向提问 | 其他 |

VLM 裁决三原则（E7 实测）：① 像素证据充分的直判、不问 VLM；② 问 VLM 时带上 8 连续帧拼图 +
触发信号摘要 + intended prompt，问具体问题（"主体逐格是否连续""出画是否有意"），别问"有没有问题"；
③ **rubric 先行**：看视频前先由 judge 从 prompt 生成 3-5 条硬标准附在裁决 prompt 里（Nova 1/5 →
Claude 4/5 → Claude+rubric 5/5）。

## 3. Wan 资产（`/home/ec2-user/efs/wan/`）

- 代码：`Wan2.1/`、`Wan2.2/`（各自带 generate.py）；`DiffSynth-Studio/`、`LightX2V/` 备选加速方案
- 权重：`Wan2.1-T2V-1.3B`（轻）、`Wan2.2-T2V-A14B`（重，MoE）、`Wan2.2-TI2V-5B`（中）
- 参考：`run_benchmark.py` 展示了调用方式（`/opt/pytorch/bin/python3 generate.py ...`，cwd 在代码目录）;
  `benchmark_results*.md` 有各配置的耗时/显存数据，**先读它避免重新踩坑**
- 5s 视频:Wan 默认 16fps → 81 帧左右;分辨率按 benchmark 里验证过的配置来

## 4. 要构建的东西

### 4.1 视频生成 Agent（管线可以是脚本编排,"agent"体现在 LLM 决策环节）

```
input prompt（用户一句话）
   │
   ▼
[导演 Agent]  用 Bedrock Claude Sonnet 4.5 把 input 扩写成 3 个分镜（shot list）：
   │          每个分镜输出 {shot_id, wan_prompt(英文,含镜头/风格描述), duration≈5s,
   │          expected_subjects[], camera, transition_to_next(cut/fade/dissolve)}
   │          ——这个 JSON 同时就是评估用的 intended prompt + 分镜元数据!
   ▼
[3 × 生成 Agent]  每个分镜分别用 Wan2.1 和 Wan2.2 生成 5s 视频（同 seed 策略自定）
   ▼
[拼接]  ffmpeg 按 transition_to_next 拼接（fade 用 xfade filter,cut 直接 concat）
   ▼
成片 + 分镜元数据 JSON（转场时间点必须记录——评估时豁免 T4）
```

### 4.2 五个测试 prompt

客户关心"常见生成场景"。⚠ 一条合规建议：**避免真实公众人物 + 暴力/武装场景**（如"特朗普持枪战斗"
这类,真人+虚构暴力内容有 deepfake/信息误导风险,客户平台上架也会遇到同样审核问题）。
用虚构角色保留同等技术难度（单人+装备+动作+多镜头连续性）。建议五条（可自行调整）：

1. **动作**：一名穿防弹背心的特种兵在废墟中持枪突进,烟雾弥漫,电影感（考验:人物一致性/装备细节/动作连贯）
2. **动物**：一只柯基在雪地里追飞盘,慢镜头,阳光（考验:运动平滑/物理合理性）
3. **产品**：一瓶饮料在旋转展台上,水珠滑落,棚拍打光（考验:刚体稳定/文字标签/闪烁）
4. **人文**：雨夜霓虹街头,一位撑伞行人走过积水倒影,赛博朋克（考验:光影一致/背景连续）
5. **多主体**：两名击剑运动员对决,白色剑服,体育馆(考验:多主体身份不混淆/交互合理)

每条 → 导演 3 分镜 → Wan2.1 与 Wan2.2 各生成 → 各拼一条成片。共 10 条成片 + 30 段分镜 clip。

### 4.3 评估对接

对每条成片（和可选每段 clip）跑：
1. `fast_scan.scan()` 拿全帧信号（GPU 上秒级）
2. 按 §2 阈值表跑 detector 直判(注意:转场点豁免 T4;生成视频冻结判定恢复直判)
3. 候选窗 + 主体窗 → Claude+rubric 裁决(抄 e7_claude_verdict.py)
4. T10:成片 8 帧 vs 导演分镜的 intended prompt 对齐评分
5. 汇总输出（前端直接消费）:
```json
{"video": "...", "model": "wan2.1|wan2.2", "prompt_id": 1,
 "shots": [...导演分镜元数据...], "transitions_s": [5.0, 10.0],
 "findings": [{"type","start_s","end_s","severity","evidence","confidence","verdict_by"}],
 "signals_preview": {...降采样后的信号曲线,画图用...},
 "scores": {"per_dim": {...}, "hard_fails": [...], "total": 0-100},
 "wan21_vs_wan22": "同 prompt 对比要能在前端并排看"}
```

### 4.4 前端展示平台（GitHub Pages）

- 仓库 **caitq2024/video-eval-agent** 已创建（CPU 机器推的初始版,含管线代码+文档+`docs/` 脚手架）,
  Pages 从 `docs/` 目录发布,风格参考 caitq2024.github.io/auto_auction（纯静态 + data.json 驱动）
- 页面建议：① 首页:5 个 prompt 卡片×2 模型,缩略图+总分;② 详情页:成片播放器 + 信号
  时间轴图（canvas/echarts 画 signals_preview,缺陷窗口高亮红块,点击跳转视频时间）+ findings 列表
  （类型/时间/证据/裁决方）+ 三个分镜 clip + 导演分镜 JSON;③ 对比页:wan2.1 vs wan2.2 并排;
  ④ 方法页:把 TAXONOMY.md 的 10 类表格渲染出来
- 视频文件放 `docs/assets/videos/`（5s 480p 每条几 MB,30+10 条可控;若超 100MB 考虑压码率,
  GitHub 单文件限 100MB、仓库软限 1GB）
- 部署：push 到 main 即生效（Pages 已配 docs/）;本地预览 `python3 -m http.server -d docs`

## 5. 注意事项 / 已知坑

1. **阈值是在 640×360 合成视频上校准的**。Wan 生成视频纹理噪声更大,先跑 1-2 条 clean 生成视频
   看各信号底噪,必要时按「clean 校准 ×1.5」重设下限(fuse 的门控哲学不变)
2. **真实生成视频里"冻结"可能是正常的静止镜头**——用导演分镜里的 camera/motion 字段做先验
3. Bedrock 裁决调用在 CPU/GPU 机器都能跑(boto3 已配)；Claude 裁决 ~$0.01-0.02/次,30 段 clip
   全量精查也就几美元
4. Wan2.2-A14B 显存大,先用 TI2V-5B 或 2.1-1.3B 打通全流程再换大模型;LightX2V 可提速
5. 生成失败/超时也是"评估素材"——T0 文件层 gate(ffprobe)记得跑
6. 前端别用需要 build 的框架,纯 HTML+JS(参考 auto_auction),GPU 机器改完直接 push
7. push 用 CPU 机器的 gh 登录(EFS 共享,直接在仓库目录 git push 即可;若 GPU 机器没凭证,
   把产物写进 EFS 仓库目录后提醒用户或让 CPU 侧会话推)

## 6. GitHub 仓库结构（已建）

```
video-eval-agent/
├── README.md              项目说明
├── HANDOFF_WAN_EVAL.md    本文档副本
├── pipeline/              评估管线代码（experiments/scripts 精选副本）
├── taxonomy/              TAXONOMY.md + REPORT*.md + e1*/e7 报告
├── docs/                  GitHub Pages 站点（index.html 脚手架 + data/ + assets/videos/）
└── wan_agent/             ← 你要写的:导演/生成/拼接/评估编排代码放这里
```

本地克隆已放在 `/home/ec2-user/efs/agent_evaluation/video_eval/repo/`（EFS,两台机器都能直接用）。

## 7. 验收清单

- [ ] 5 prompt × 2 模型 = 10 成片(每条 3 分镜拼接,含转场),30 clip
- [ ] 每条成片一份评估 JSON(findings 带时间窗与证据;转场点未被误报为 T4)
- [ ] wan2.1 vs wan2.2 同 prompt 对比结论(哪个模型哪类缺陷多)
- [ ] GitHub Pages 可访问,视频可播放,时间轴可交互
- [ ] 把新发现(生成视频上的阈值调整、Wan 特有缺陷模式)追加到 results/REPORT_R2.md
