# 交接文档：MiniMax-H3 视频生成基座测试（A100 → H200 机器）

写给 H200 机器上的 Claude Code。日期 2026-08-17。
上一台机器：8×A100-SXM4-**40GB**（H3 权重驻留放不下，故迁移）。
**先通读本文档再动手。** 环境初始化：`source /home/ec2-user/efs/claude_code.sh`
（Claude Code 走 Bedrock，裁决模型 `global.anthropic.claude-opus-4-8`）。
EFS `/home/ec2-user/efs` 跨机器共享，所有代码/权重/产物直接可用。

---

## 0. 一句话背景

我们给客户做「视频生成 Agent 质量评估」PoC，已完成完整闭环：
**导演 Agent（Claude 拆分镜）→ Wan2.1/2.2 生成 → ffmpeg 转场拼接 → 20 类缺陷评估
（探针定位 + Hybrid 裁决）→ GitHub Pages 可视化**。10 个测试 case × 2 个 Wan 模型
= 60 clip + 20 成片全部生成并评估完毕。
**当前任务**：客户要试更强的生成基座 **MiniMax-H3**（33B 音视频一体生成模型），
把 10 个测试 case 用 H3 出一版，跑同一套评估管线，与 Wan 对比。
准备工作已全部就绪（见 §4），只差「起服务 → 生成 → 评估」——A100 40GB 卡在显存上。

## 1. 必读材料（按顺序）

| 文档 | 内容 |
|---|---|
| `video_eval/HANDOFF_WAN_EVAL.md` | 第一份交接：评估方法论、Wan 资产、阈值体系（部分已被 v2 迭代超越，读背景即可） |
| `video_eval/experiments/results/REPORT_R2.md` | **R1-R9 全部实验记录**：R5 真实视频阈值修正、R5.2 跟踪器否决 VLM 幻觉、R7 v2 第一批、R8 T12-T16 落地与 p4 雨伞验收、R9 换装/故事性 case |
| `video_eval/experiments/results/TAXONOMY_V2_proposal.md` | 20 类缺陷完整定义、文献依据、检测方案 |
| `video_eval/repo/wan_agent/*.py` | 全部编排代码（见 §2.3 文件地图） |
| 本文档 §5 | H3 已完成的准备与 A100 踩坑记录 |

## 2. 已完成工作全景

### 2.1 视频生成 Agent（repo/wan_agent/）

```
一句话创意（中文）
  → director.py   导演 Agent（Bedrock Claude）：拆 3 分镜 JSON（~11s）
                  {wan_prompt(英文40-80词), expected_subjects[], camera,
                   motion_level, duration_s=5, transition_to_next}
                  ★ 分镜 JSON 同时是评估输入：转场豁免/主体跟踪词/运动先验/intended prompt
  → generate_clips.py  多卡并行派发 Wan generate.py（GPU 池+队列）
  → stitch.py     ffmpeg 拼接：cut→concat / fade|dissolve→xfade(0.5s)
                  ⚠ 每级 concat/xfade 输出必须补 ,fps=N（CFR 声明会丢）
                  转场时间点写入 shots.json['films'][model]['transitions']
```
- 测试集：**10 个 case p1-p10**（特种兵/柯基/汽水/霓虹雨夜/击剑/招牌文字/篮球物理/
  环绕老人/模特换装/便利店故事），ideas*.json 三个文件
- 产物结构：`wan_outputs/<pid>/{shots.json, <model>/{shot1..3.mp4, film.mp4, eval.json}}`
  model ∈ {wan2.1, wan2.2, **minimax-h3**(待生成)}
- Wan 参考耗时（A100）：2.1-1.3B ≈289s/段、2.2-5B ≈497s/段（offload，不加 --t5_cpu）

### 2.2 评估 Agent（20 类 v2，repo/wan_agent/evaluate.py + probes_v2.py）

```
T0 ffprobe gate → S1 fast_scan 全帧信号(GPU,0.7s/15s视频：亮度/帧差/闪烁/CLIP/
光流warp+8×8块残差+相机流) → S2 直判(9类,~0s) → S3 GroundingDINO 主体探针(6s)
→ S3b v2 探针(KeypointRCNN 骨长/腕部、交互区域、文字提取,0.8s)
→ S4/S5 Claude 裁决(≈22 次调用 16 路并行,~25s) → eval.json
```
- **不依赖 Bedrock 的 9 类直判**：T0/T1/T2/T3/T4/T6/T16/T17/T20（--no-vlm 模式 ~9s/片）
- **VLM/LLM 的 11 类**：T5/T7/T8/T9/T10/T11/T12/T13/T14/T15/T19（+T20 改写忠实度）
- 关键机制（都来自真实误报复盘，别退化）：
  ① 转场豁免双侧（探针 mask + VLM prompt 附转场时间表）② 主体消失 GroundingDINO
  主判、单实例 ≥95% 检出率否决 VLM 幻觉（双重验证标 dual）③ T10 三票取中位
  ④ T1 孤立尖峰+局部中位相对化、T7/T3/T17 按 motion_level 缩放 ⑤ T12 严格证据规则
  （剪影手指不可见≠缺陷，物体级结构异常仍可判）
- 全部输出中文（detector 证据/VLM 理由/rubric/T10 评语）；~33s/片
- 20 成片终版分数：wan2.1 均值≈52 / wan2.2≈59；T19 跨镜头一致性是最高产新类（9 处）

### 2.3 代码文件地图（repo/wan_agent/）

| 文件 | 职责 |
|---|---|
| common.py | 路径/MODELS 注册表(含 minimax-h3)/ask_claude(Bedrock+重试) |
| director.py / ideas*.json | 导演 Agent 与 10 个测试创意 |
| generate_clips.py / stitch.py / run_pipeline.py | Wan 生成与拼接编排 |
| **h3_promptify.py** | 把现有分镜转写成 H3 T2VA 结构 prompt（已跑完） |
| **generate_h3.py** | H3 生成客户端（HTTP→SGLang），生成+自动拼接 |
| evaluate.py / probes_v2.py | 20 类评估管线 |
| run_all_evals.py | 批量评估（已含 minimax-h3；跳过 demo_*） |
| build_site.py | 聚合 data.json + 视频转码到 docs/（已含 minimax-h3） |
| demo_server.py + demo.html | 现场 demo：prompt 生成 + ≤50MB 视频上传评估（端口 8008） |

### 2.4 展示与汇报

- GitHub Pages：https://caitq2024.github.io/video-eval-agent/（repo 本地克隆
  `video_eval/repo/`，**GPU 机器无 git 凭证，push 要么装 gh 登录 caitq2024，
  要么在 CPU 机器推**）
- PPT v2：`video_eval/视频生成Agent质量评估_汇报_v2.pptx`（45 页，构建脚本
  pptx_build/build_v2.py；模板是深底 #161D26 白字 Amazon Ember，别按白底做）

## 3. 当前任务：MiniMax-H3 基座测试

**目标**：10 个 case 用 H3 出一版（30 clip + 10 成片），跑同一套评估，与 Wan2.1/2.2
三方对比，前端与报告同步更新。

### 3.1 H3 是什么（调研结论）

- 33B **dense** 单流 Omni-Transformer，**音视频一体生成**（4-15s、24fps、短边 768、
  32kHz 立体声）；CFG-distilled bf16；文本编码器是 **Qwen3-VL-32B**（这就是显存大头）
- 权重已下载：`/home/ec2-user/efs/base_model/MiniMax-H3/`（FL2VA 变体 135GB，
  含 transformer/text_encoder/video_vae/audio_vae/processor/tokenizer + model_index.json）
- T2VA（纯文生视频）= FL2VA 变体无图模式
- 官方推理：SGLang `sglang serve --model-variant fl2va`（venv 装 `sglang[all]`
  得 0.5.17 已支持 H3）；备选 diffusers ModularPipeline
- API：`POST /v1/videos {task:'t2va', prompt, conditions:[], target:{short_edge:768,
  aspect_ratio:'16:9', duration_seconds:5}, seed}` → 轮询 `GET /v1/videos/{id}`
  → `GET /v1/videos/{id}/content` 下载 mp4。generate_h3.py 已封装
- **prompt 规范**（官方 skills/h3-prompt-writing）：三字段纯文本 ——
  `integrated_multimodal_description:`（[Shot 1] 开头、风格+构图+主体+动作时间线+
  运镜三维度句式；画面文字用双引号）`overall_soundscape:`（1-3 句环境音）
  `non_diegetic_music:`（1-2 句配乐，禁抽象情绪词）

### 3.2 已完成的准备（直接可用）

1. **30 条 h3_prompt 已生成**：h3_promptify.py 用 Claude 按官方规范把 p1-p10 每镜
   wan_prompt 转写为 T2VA 三字段格式，存于各 `wan_outputs/<pid>/shots.json` 的
   `shots[k]['h3_prompt']`。**复用同一套分镜与 seed（4200+shot_id）→ 与 Wan 公平对比**
2. **generate_h3.py** 客户端写好（提交/轮询/下载/写 generation 元数据/自动拼接）
3. minimax-h3 已注册进 common.MODELS（fps=24）、run_all_evals、build_site（含 label）
4. evaluate.py 对非 Wan 模型的 T20 规格检查已做兼容（MODELS.get）

### 3.3 A100 40GB 踩坑记录（H200 上大概率不会遇到）

| 尝试 | 结果 |
|---|---|
| 官方配方 4 卡 `--performance-mode speed` | 加载 OOM（权重每卡复制：TE 32GB 驻留 + DiT 装不下） |
| `--performance-mode memory`（4 卡/8 卡） | 服务能起但**运行时 OOM**（驻留权重把 40GB 打满，激活无余量） |
| `--use-fsdp-inference + --text-encoder-cpu-offload + --vae-cpu-offload` | 加载成功（每卡仅 15GB），但生成时报 **"Expected all tensors to be on the same device"**——encoder-cpu-offload 与 FSDP 组合的 bug |
| 纯 `--use-fsdp-inference`（8 卡，无 offload 旗标） | **未验证完**（迁移前最后一个候选配置，H200 上可作为备选） |

其他备忘：JIT fused QKNorm+RoPE kernel 需要 `ninja`（pip 装进 venv 即可，否则回退慢核）；
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 建议带上。

### 3.4 H200 上的启动建议（按顺序尝试）

H200 141GB 单卡：TE 64GB + DiT ~40GB + VAE + 激活 ≈ 单卡都可能放下，多卡只为提速。

```bash
# 1) venv（本地盘，不要放 EFS）
/usr/bin/python3 -m venv ~/venvs/sglang   # 或机器自带的 python3.10+
~/venvs/sglang/bin/pip install "sglang[all]" ninja

# 2) 首选：官方配方（H200 显存充裕）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
~/venvs/sglang/bin/sglang serve \
  --model-path /home/ec2-user/efs/base_model/MiniMax-H3 \
  --num-gpus 4 --ulysses-degree 4 --performance-mode speed \
  --host 0.0.0.0 --port 30010 --model-variant fl2va
# 卡数按机器实际；单卡也可先 --num-gpus 1 冒烟
# 若仍紧张 → 加 --use-fsdp-inference（A100 验证过加载最省）；别用 encoder-cpu-offload（有 bug）

# 3) 冒烟（p6 招牌镜头，看单段耗时）
cd /home/ec2-user/efs/agent_evaluation/video_eval/repo/wan_agent
/opt/pytorch/bin/python3 -c "
import json; from generate_h3 import gen_one
m=json.load(open('/home/ec2-user/efs/agent_evaluation/video_eval/wan_outputs/p6/shots.json'))
print(gen_one('http://localhost:30010', m['shots'][0]['h3_prompt'], '/tmp/h3_smoke.mp4', 4201, 5))"
# ⚠ 本机 python 路径可能不同：任何有 requests/numpy 的 py3.10+ 都行
```

## 4. 接下来要做的事（step by step）

1. **起服务 + 冒烟**（§3.4）；记录单段 5s 的墙钟耗时（写进报告）
2. **全量生成 30 clip**：
   `python3 generate_h3.py --concurrency 2`（并发数按服务吞吐调；服务端内部排队）
   —— 会自动写 generation 元数据并对每个 pid 拼接 film.mp4（24fps）
   ⚠ 拼接只处理视频轨，**H3 的音频轨会被丢弃**（评估不涉及音频）；分镜 clip 本身
   带声音，可在前端/汇报中作为 H3 特色展示；有余力可给 stitch 加音频 concat
3. **评估**：`HF_HOME=~/hf_cache TORCH_HOME=~/torch_cache python3 run_all_evals.py --device cuda:0`
   （无 --force 只评新增的 minimax-h3；评估进程需要 /opt/pytorch 同款依赖：torch/
   opencv/open_clip/transformers/torchvision —— **评估用回原 A100 机器也完全可以**，
   EFS 共享，产物互通）
4. **对比分析**：三模型分数矩阵 + 缺陷类型分布对比（Wan 的顽疾 T12 手部/T19 跨镜头/
   T13 物理/T15 文字，H3 是否更好？H3 特有缺陷模式？）；关注 p5 击剑（多主体）、
   p7 篮球（物理）、p6 招牌（文字）、p10 便利店（叙事）这几个 Wan 重灾 case
5. **前端**：`python3 build_site.py`（自动转码 H3 视频进 docs/assets/videos/，
   data.json 已支持第三模型）；前端对比页目前是双模型布局，若要三方并排需小改
   compare.html（可选）
6. **报告**：REPORT_R2.md 追加 R10 节（H3 部署踩坑 + 生成耗时 + 三方对比结论）
7. **push**：CPU 机器（gh 已登录）`cd repo && git push`；或本机 `gh auth login` 后推
8. 可选加分项：H3 支持单 prompt 多 [Shot] 一体生成 15s（自带镜头切换与连贯音频）——
   挑 1-2 个 case 试试「一体生成 vs 逐镜拼接」的对比（评估时注意：一体生成没有转场
   元数据，T4 会把模型自己的切镜标为意外切换，需在 shots.json 里造 transitions 或说明）

## 5. 环境备忘（新机器自查清单）

- [ ] EFS 挂载在 `/home/ec2-user/efs`（所有路径硬编码以此为前提）
- [ ] `source /home/ec2-user/efs/claude_code.sh`（Bedrock 凭证与模型选择在里面）
- [ ] Bedrock us-west-2 可用（评估裁决用 `global.anthropic.claude-opus-4-8`，
  **该模型不接受 temperature 参数**，common.py 已适配）
- [ ] HF_HOME / TORCH_HOME 指向本地盘（评估要下 GroundingDINO/KeypointRCNN/RAFT/CLIP）
- [ ] ffmpeg 用共享静态版 `video_eval/bin/ffmpeg`（系统可能没装）
- [ ] 评估侧 python 需要：torch+cuda、opencv-headless、open_clip_torch、transformers、
  torchvision、boto3、requests（A100 机器用的 /opt/pytorch，新机器若没有就 pip 装）
- [ ] 首跑评估会有模型下载（~1GB），fast_scan/GroundingDINO/KeypointRCNN 均有进程内缓存

## 6. 验收清单

- [ ] H3 服务起稳，单段 5s 生成耗时有记录
- [ ] 30 clip + 10 成片（`wan_outputs/p*/minimax-h3/`），generation 元数据完整
- [ ] 10 份 eval.json，评估分数与 findings 完整（中文证据）
- [ ] 三模型对比结论写入 REPORT_R2.md R10
- [ ] 前端更新并 push（三模型可见、H3 视频可播）
- [ ] 新踩的坑追加到本文档或 REPORT
