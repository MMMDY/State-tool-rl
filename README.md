# State Tool RL

面向多轮工具调用智能体的状态感知强化学习训练与评测项目。项目以
[tau2-bench](https://github.com/sierra-research/tau2-bench) 的 airline、retail、telecom
任务为环境，并使用 [SLIME](https://github.com/THUDM/slime) 训练基础设施。

## 核心思路

多轮任务的最终成败通常只在结束时给出。State Tool RL 从环境状态检查中提取动作、沟通
和环境断言的完成度，构造更密集的训练信号：

```text
shaped_reward = task_reward + alpha(domain) × partial_score
```

随后在同一 prompt 的多条 rollout 上计算 GRPO 优势；课程策略会降低已解决和极难任务的
权重。Telecom 场景会提高沟通维度的权重，以适应由用户执行诊断操作的 dual-control 约束。

## 功能

- 多轮 rollout：将 Qwen 工具调用转换为 tau2 环境动作，格式错误时会进行一次修复。
- 状态感知奖励：解析动作、沟通、环境断言和数据库检查，形成部分得分。
- 领域自适应与课程采样：奖励系数、部分得分权重和任务权重均可配置。
- 双环境运行：SGLang 服务与 tau2/SLIME 训练环境分离，避免依赖冲突。
- 可恢复评测：每个 task 和 trajectory 原子落盘，重跑可复用已完成结果。
- 可审计报告：记录采样、thinking、完成原因、轨迹和奖励组成，并脱敏凭证字段。

## 项目结构

```text
configs/                  policy 与用户模拟器配置
eval/eval_passk.py        SGLang policy 的 tau2 Pass@k 评测器
requirements/             Python 3.12 / CUDA 12.8 锁定依赖与安装说明
scripts/00_doctor.sh      环境、GPU、依赖、端口与子模块检查
scripts/start_policy.sh   启动待评测的 policy 服务
scripts/start_user_sim.sh 启动训练用用户模拟器
scripts/train_sft.sh      SFT 训练入口
scripts/train_grpo.sh     GRPO 训练入口
tau2_rl_pipeline/         环境适配、动作解析、prompt、reward、rollout、任务索引
EVAL.md                   已登记的评测基线
```

## 环境准备

目标环境为 Linux、Python 3.12、CUDA 12.8 与 NVIDIA GPU。完整的原生安装步骤与锁定依赖
见 [requirements/README.md](requirements/README.md)。克隆时初始化固定版本子模块：

```bash
git clone --recurse-submodules https://github.com/MMMDY/State-tool-rl.git
cd State-tool-rl
git submodule update --init --recursive
```

项目使用两套虚拟环境：`.venv` 仅承载 SGLang policy/user-simulator 服务；`.venv-tau2`
承载 tau2、Ray、SLIME 和 Megatron 训练。不要混用两套依赖。

复制并填写配置（使用云端用户模拟器或 Judge 时需填写 API key）：

```bash
cp configs/.env.example configs/.env
set -a && source configs/.env && set +a
bash scripts/00_doctor.sh
```

环境检查结果会写入 `reports/doctor/latest.json`。若路径不同，可用 `TAU2_PYTHON` 和
`SGLANG_PYTHON` 覆盖两套 Python 解释器路径。

## 快速评测

先启动待评测的 policy 服务：

```bash
MODEL_DIR=/path/to/checkpoint GPUS=0 TP=1 bash scripts/start_policy.sh
```

在另一个终端执行评测：

```bash
.venv-tau2/bin/python eval/eval_passk.py \
  --sglang-url http://127.0.0.1:30000 \
  --sglang-model qwen3-4b \
  --domains airline,retail,telecom \
  --task-split test \
  --num-samples 4 \
  --output outputs/eval_test_k4.json
```

默认 policy profile 位于 [configs/qwen3-4b.yaml](configs/qwen3-4b.yaml)，默认启用
thinking；加 `--no-enable-thinking` 使用非 thinking 配置。用户模拟器配置位于
[configs/simulator.yaml](configs/simulator.yaml)。

输出结构如下：

```text
outputs/eval_test_k4/
├── eval_test_k4.json     # 汇总报告，仅全部完成后生成
├── checkpoint.json       # 可恢复状态
├── task_results/         # 每个任务的原子结果
└── trajectories/         # 每个 sample 的紧凑轨迹
```

`Pass@k` 使用 tau2 官方 `compute_metrics()` 定义；`best_of_k_success` 只是“至少一条
rollout 成功”的诊断指标，不能与 Pass@k 混用。历史评测登记在 [EVAL.md](EVAL.md)。

## 训练流程

### 1. 准备任务

生成训练 task 索引：

```bash
.venv-tau2/bin/python tau2_rl_pipeline/tasks.py \
  --local_dir outputs/tau2/tasks \
  --domains airline,retail,telecom \
  --splits train
```

### 2. SFT

设置 `HF_DIR`、`TORCH_DIST_DIR`、`SFT_DATA_JSONL` 等路径后运行：

```bash
NUM_GPUS=4 bash scripts/train_sft.sh
```

脚本以 Qwen chat template 和多轮 loss mask 训练工具调用格式与交互协议。

### 3. GRPO

在与训练 GPU 分开的设备上启动本地用户模拟器：

```bash
MODEL_DIR=/path/to/user-model GPUS=2 TP=1 bash scripts/start_user_sim.sh
```

再启动 GRPO：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/train_grpo.sh
```

常用变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TAU2_REWARD_ALPHA` | `0.25` | 部分得分的基础奖励系数 |
| `TAU2_DOMAIN_ADAPTIVE_ALPHA` | `1` | 是否按领域调整奖励系数 |
| `TAU2_USE_CURRICULUM` | `1` | 是否启用任务课程权重 |
| `TAU2_CURRICULUM_MIN_ATTEMPTS` | `5` | 开始调整权重前的最小尝试数 |
| `TAU2_MAX_STEPS` | `100` | 单条 rollout 最大交互步数 |
| `TAU2_USER_API_BASE` | `http://127.0.0.1:30001/v1` | 用户模拟器端点 |

完整训练参数见 [scripts/train_grpo.sh](scripts/train_grpo.sh)。

## 排障

- 先运行 `bash scripts/00_doctor.sh`，确认子模块、两套 Python、CUDA、GPU 和端口正常。
- SGLang 显存不足时，降低 `MEM_FRACTION` 或 `--max-tokens-per-gpu`，并减少并发请求数。
- 评测中断后，以相同的 `--output` 路径再次运行即可恢复，不要删除输出目录。
- policy、用户模拟器与 Ray 的 GPU 应互不重叠；检查 `GPUS` 和 `CUDA_VISIBLE_DEVICES`。
- `TAU2_CLEANUP=1` 会停止本机 SGLang/Ray 进程，仅在确认没有其他任务使用它们时启用。

## 致谢与许可证

本项目使用 [tau2-bench](https://github.com/sierra-research/tau2-bench)、
[SLIME](https://github.com/THUDM/slime) 与 Megatron-LM。原项目说明已保留在
[README.backup.md](README.backup.md) 供追溯。

本项目采用 [Apache-2.0](LICENSE) 许可证。
