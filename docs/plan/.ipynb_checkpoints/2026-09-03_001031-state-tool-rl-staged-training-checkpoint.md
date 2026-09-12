# StateTool-RL 分阶段训练实施计划（基于 Tau2-RL-Pipeline / SLIME）

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在 `jbarnes850/Tau2-RL-Pipeline` 的代码基础上，完成「官方 base 评测 → 环境审计 → SFT → 离线 Step-RFT → terminal-only GRPO → StateTool-RL」的可复现 τ²-Bench 多轮工具调用训练闭环，并在严格 held-out 任务上报告成功、成本、错误与违规指标。

**Architecture:** Fork 上游项目并固定 commit；沿用 `tau2_rl_pipeline/actions.py`、`prompting.py`、`env.py`、`rollout.py`、`tasks.py` 和 SLIME 的 multi-turn loss mask / GRPO hook。训练底座唯一为 **SLIME + SGLang + Ray/Megatron-LM**：SFT 使用 SLIME `train_async.py`，GRPO 使用 SLIME `train.py`，SGLang 负责 policy rollout，Ray/Megatron-LM 负责分布式执行。新代码只扩展官方评测、轨迹记录、环境状态审计、step verifier、offline candidate dataset 和两个 reward hook；**不引入或维护 verl、vLLM/FSDP2、第二套 PPO/GRPO trainer。**

**Tech Stack:** Python 3.12、PyTorch、Transformers、SLIME（pin SHA）、Megatron-LM（pin SHA）、Ray、SGLang、τ²-Bench（pin SHA）、Gymnasium、Docker（可选隔离）、W&B/TensorBoard。

## 总流程图

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 固定输入：upstream SHA / SLIME SHA / Megatron SHA / τ² SHA / 模型版本 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P0  Benchmark + LLM API Base Evaluation                              │
│ τ² airline manifest → policy API → AgentGymEnv/tool loop             │
│ → official evaluator → rollouts.jsonl → avg_reward / any-success@K   │
└────────────────────────────────┬────────────────────────────────────┘
                  P0 gate：可重算、可追溯、无训练
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P1  Environment Audit                                                 │
│ reset/seed → snapshot/state hash → replay → 8-way parallel isolation │
└────────────────────────────────┬────────────────────────────────────┘
                  P1 gate：无状态污染、异常不崩溃
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P2  Data Protocol + Prompt Baseline                                  │
│ task registry → template/domain split manifests → SGLang baseline    │
└────────────────────────────────┬────────────────────────────────────┘
                  P2 gate：test 隔离、prompt 基线冻结
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
┌───────────────────────────┐              ┌──────────────────────────┐
│ P3  Multi-source SFT      │              │ train-only trajectories  │
│ demo + teacher + repair   │─────────────▶│ gold / teacher / switch  │
│ SLIME train_async.py      │              │ loss-mask audited        │
└─────────────┬─────────────┘              └──────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P4  Offline Step-RFT                                                  │
│ SFT prefix → K candidates → format/action/args/progress verifier     │
│ → weighted-RFT smoke → SLIME custom group-step objective              │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P5  Fair Online RL Baseline                                           │
│ terminal official reward only → SLIME GRPO + SGLang rollout          │
│ (alpha=0, curriculum=off, partial score=off)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P6  StateTool-RL                                                       │
│ same training topology; only reward hook differs                      │
│ terminal + state progress − tool cost − invalid − unsafe behavior     │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ P7  Frozen Held-out Evaluation                                        │
│ test-id/test-ood + ToolSandbox → CI / ablation / failure taxonomy     │
└─────────────────────────────────────────────────────────────────────┘

关键不变量：P0–P7 均复用同一 action parser、prompt contract、user simulator
与 SLIME + SGLang + Ray/Megatron 训练链路；test 永不参与训练或 reward 设计。
```

---

## 0. 范围、研究问题与不可妥协约束

训练目标不是生成“看起来像 JSON 的 function call”，而是在多轮业务环境中正确读取 observation、选择工具、填写参数、处理状态变化、遵守 policy、必要时澄清并及时终止。

研究问题：

> 将官方终局 verifier 衍生为不泄漏的状态/步骤信号，并结合失败—恢复数据，能否改善多轮 Tool Agent 的 long-horizon credit assignment，同时不增加无效调用、错误副作用或 policy violation？

第一版不做：

- 不训练 32B/72B；先以 4B–9B 模型验证闭环；
- 不接真实支付、CRM 或生产 API；
- 不在 test 集上选择 prompt、reward 权重、数据配比或训练步数；
- 不把 reference trajectory 当作唯一正确动作序列；
- 不把 user message、tool observation 或 hidden reference target 纳入 SFT policy loss；
- 不在环境隔离、日志和官方评测未通过前启动大规模 RL；
- **不并存第二套训练框架。**

---

## 1. Fork、目录、依赖和运行时契约

### 1.1 固定上游并创建 fork

固定当前上游 SHA，并在实验登记表中保存它：

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/jbarnes850/Tau2-RL-Pipeline.git state-tool-rl
cd state-tool-rl
git remote rename origin upstream
git checkout -b state-tool-rl/<PINNED_UPSTREAM_SHA> <PINNED_UPSTREAM_SHA>
git remote add origin git@github.com:<YOUR_GITHUB_ID>/state-tool-rl.git
git push -u origin HEAD
```

目录以 upstream package 为核心：

```text
state-tool-rl/
├── configs/
│   ├── environment/tau2_airline.yaml
│   ├── data/{split_v1.yaml,mixture_v1.yaml}
│   └── training/{sft.env,step_rft.env,terminal_grpo.env,state_tool_grpo.env}
├── data/{eval_manifests,splits,processed}/                 # gitignore 原始轨迹
├── eval/
│   ├── eval_base.py                                        # 从 eval_passk.py 重构
│   └── eval_passk.py                                       # 兼容入口
├── scripts/
│   ├── 00_doctor.sh
│   ├── 01_smoke_tau2.py
│   ├── 02_make_splits.py
│   ├── 03_collect_rollouts.py
│   ├── 04_build_sft_data.py
│   ├── train_sft.sh                                        # 参数化 upstream 脚本
│   ├── 06_train_step_rft.sh
│   ├── 07_train_terminal_grpo.sh
│   ├── 08_train_state_tool_grpo.sh
│   └── 09_reproduce_main_table.sh
├── tau2_rl_pipeline/
│   ├── actions.py, prompting.py, env.py, rollout.py, tasks.py
│   ├── reward_terminal.py, reward_state_tool.py
│   ├── evaluation/{runner,metrics,failure_analysis}.py
│   ├── data/{split,candidate_collector,step_dataset,switch_policy}.py
│   ├── verifiers/{format,action,argument,progress,aggregate}.py
│   └── utils/{seed,state_hash,logging}.py
├── tests/{envs,data,verifiers,rewards,integration}/
├── artifacts/                                              # gitignore
└── reports/{experiment_registry.csv,main_results.md,failure_cases.md}
```

### 1.2 依赖与路径

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'

# 三个外部仓库均锁定 SHA；训练脚本只读取这些变量。
export SLIME_DIR="$PWD/third_party/slime"
export MEGATRON_LM_DIR="$PWD/third_party/Megatron-LM"
export TAU2_REPO_DIR="$PWD/third_party/tau2-bench"
export HF_HOME="$PWD/artifacts/hf_cache"
export TOKENIZERS_PARALLELISM=false
bash scripts/00_doctor.sh
```

`00_doctor.sh` 必须记录：fork upstream SHA、SLIME SHA、Megatron-LM SHA、τ²-Bench SHA、Python/CUDA/PyTorch/Ray/SGLang/Transformers 版本，以及 GPU、磁盘、共享内存、端口。所有脚本禁止依赖 `/root/slime`、`/root/Megatron-LM` 等绝对路径。

### 1.3 复用与改造边界

| Upstream 文件 | 复用内容 | 必须改造 |
|---|---|---|
| `actions.py` | Qwen 风格单 action parser、环境动作转换、observation follow-up | 输出规范化错误码；P0/P1 不能静默丢弃 malformed action |
| `prompting.py` | 根据 task 动态插入 policy 与 tool schema | 保存 system prompt/chat-template hash；compressed prompt 只作为单独消融 |
| `eval/eval_passk.py` | `AgentGymEnv` 控制流、官方终局 reward 读取 | 重构为通用 `eval_base.py`；新增 API client、manifest、seed、trajectory 和汇总 |
| `env.py` | `reward_info` 与 action/communication/env/DB 子项解析 | 新增 snapshot、state hash、state diff、reset audit |
| `rollout.py` | SLIME custom generate、多轮 messages、loss mask | 接入 `TrajectoryRecorder`；逐 step 保存 action/observation/info/state |
| `reward.py` | SLIME custom reward post-process、GRPO group normalization | 拆为 terminal-only 和 state-tool；禁用默认 curriculum |
| `train_sft.sh` / `train_grpo.sh` | SLIME/Ray/Megatron 启动骨架 | 参数化路径/seed/run dir；拆出三种明确训练脚本 |
| `tasks.py` | τ² registry task index | 按 template/domain 创建并冻结 train/dev/test manifest |

---

## 2. Phase 0：官方 Benchmark + LLM API + Airline Base 评测

**目标：** 不训练、不改 reward，先在官方 τ²-Bench `airline` 上将 LLM API、agent、tool、user simulator 和官方 terminal evaluator 跑通；产出后续全部实验的冻结 base baseline。

### 2.1 改造 `eval/eval_passk.py`

现有脚本只能请求 SGLang `/generate`，只输出 `pass_at_1/pass_at_k`，不持久化完整 trajectory，且没有均值 reward 或失败任务报告。因此：

1. 抽取 `_run_one_attempt()` 到 `tau2_rl_pipeline/evaluation/runner.py`；
2. 定义 policy client 协议，实现 `SGLangPolicyClient` 与 `OpenAICompatiblePolicyClient`；API key 仅从 `OPENAI_API_KEY` 读取；
3. 新建 `eval/eval_base.py`；`eval_passk.py` 只做向后兼容调用；
4. 引入 `data/eval_manifests/airline_smoke_v1.jsonl`，显式列出 10–20 个官方 airline task ID（查询、改签/取消、澄清、失败/禁止路径）；不得用“前 N 个 task”代替；
5. 每个 `env.step()` 后写 `TrajectoryStep`，而不是只保留最终 reward。

每条 rollout JSONL 必须保存：

```text
run_id, task_id, domain, sample_id, seed, model/provider metadata
prompt/chat-template hash, available_tools, raw_action, parsed_action
observation, safe info subset, state_before_hash, state_after_hash
terminated, truncated, wall_time_ms, token usage
official reward, official reward_info/breakdown, failure_kind, failure_step
```

不得写入 API key、Authorization header 或敏感 provider payload。

### 2.2 单任务闭环后批量 smoke

先用一个 manifest task 完成真实 API → tool → observation → API → official evaluator 闭环；API 失败、限流、空输出、格式错、未知工具均归一化为 episode failure，不能使 runner 崩溃。随后运行：

```bash
export OPENAI_BASE_URL='https://<provider>/v1'
export OPENAI_API_KEY='<SECRET>'
export ACTOR_MODEL_NAME='<BASE_MODEL>'

python eval/eval_base.py \
  --policy-backend openai --api-base "$OPENAI_BASE_URL" --model "$ACTOR_MODEL_NAME" \
  --task-manifest data/eval_manifests/airline_smoke_v1.jsonl \
  --num-samples 4 --temperature 0.0 --max-steps 16 --seed 20260903 \
  --output artifacts/evals/airline_base_smoke_seed20260903
```

`avg_reward` 是 task × sample 的官方终局 reward 均值。本文 `pass^k` 定义为同一 task 的 K 次 rollout 至少一次通过官方 success evaluator 的 task 比例；报告中同时标注它是 **any-success@K**，不是无偏 leaderboard pass@k 估计。

### 2.3 P0 报告与 Gate

生成：

```text
artifacts/evals/.../rollouts.jsonl
artifacts/evals/.../summary.json
reports/baselines/airline_base_smoke.md
reports/baselines/airline_base_failures.md
```

报告包含：`avg_reward`、success、DB success、communication success、`pass^1/pass^4`、mean/p95 tool calls、episode steps、latency、timeout/API error；列出 K 次均失败 task 的最佳 reward、失败 step、失败类型和 trajectory 路径；按成功、格式/参数、选错工具、observation 推理、policy、超步数/API 错误抽样审阅轨迹。

只有以下同时成立才进入 P1：指标能从 JSONL 重算；失败可追溯；相同配置重跑差异被记录；base report 已冻结；未启动 SFT/Step-RFT/GRPO 或 dense reward。

---

## 3. Phase 1：环境准入、状态恢复与并发隔离

**目标：** 验证“每 rollout 新建 `AgentGymEnv`”真的隔离，而非仅凭 upstream 实现假设其安全。

扩展 `tau2_rl_pipeline/env.py` 与 `evaluation/runner.py`，提供统一 `reset/step/snapshot/close` 契约；每个 step 保存 state hash、DB/state digest、消息和官方 info。实现：

```bash
python scripts/01_smoke_tau2.py --domain airline --task-id <ONE_TASK_ID> \
  --seed 20260903 --max-steps 16 --policy scripted
pytest tests/integration/test_parallel_reset.py -v -s
pytest tests/integration/test_rollout_replay.py -v -s
```

验收：同 task/seed 三次 reset 的初始 state hash 一致；相同 scripted actions 的工具返回、最终 reward/DB hash 一致；非法 action 不崩溃；完整 JSONL 能重放；至少 8 个并发 episode 不污染彼此。若上游没有可用 seed 注入点，必须记录该随机性；若共享 DB 不可隔离，则改为 process-per-episode 并限制并发。

---

## 4. Phase 2：Split 与 Prompt Baseline

**目标：** 固化无泄漏 train/dev/test 协议与可复现 prompting baseline。

用 `tau2_rl_pipeline/tasks.py` 的 registry 能力实现 `scripts/02_make_splits.py`。按 task template/entity pattern/domain 分组切分，而不是随机打散；保存 `train/dev/test_id/test_ood` manifest 和 SHA。推荐：train 60%、dev 20%、test-id 10%、test-ood 10%；训练 airline，后续扩展 retail，跨域 test 使用 telecom。

本地 policy 使用 SGLang，且评测与训练的 user simulator 必须相同：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path "$MODEL_ID" --host 127.0.0.1 --port 30000 --tp-size 2
GPUS=2,3 TAU2_USER_MODEL='<PINNED_USER_MODEL>' \
  TAU2_USER_API_BASE='http://127.0.0.1:30001/v1' bash scripts/start_user_sim.sh

python eval/eval_base.py --policy-backend sglang \
  --sglang-url http://127.0.0.1:30000/generate --hf-checkpoint "$MODEL_ID" \
  --task-manifest data/splits/v1/dev.jsonl --num-samples 1 --temperature 0.0 \
  --max-steps 16 --output artifacts/evals/prompt_dev_seed20260903
```

Gate：固定 seed/config 可复跑；registry 记录 upstream/SGLang/user simulator/split SHA；轨迹可回放；test 未用于 prompt 或 reward 调参。

---

## 5. Phase 3：多源 SFT 与 Switch-Policy 纠错数据

**目标：** 获得 tool format 稳定、可从常见错误状态恢复的 warm-start policy。

数据只来自 train split：gold/demo、teacher successful rollout、switch-policy correction。实现 `tau2_rl_pipeline/data/switch_policy.py`：weak policy 在错误前缀停止；teacher 只看公开 history/tools/policy/observation，不看 hidden gold action；仅保留“前缀错误且 teacher 修复成功/显著改善”的样本。weak-error action mask=0，teacher repair 及后续有效 action mask=1。

`04_build_sft_data.py` 生成 JSONL，复用 upstream `MultiTurnLossMaskGenerator`；system/user/tool schema/tool observation 均不参与 loss。随后：

```bash
NUM_GPUS=4 HF_DIR="$MODEL_ID" \
SFT_DATA_JSONL=data/processed/sft_balanced_train.jsonl \
SAVE_DIR=artifacts/checkpoints/sft-balanced \
SLIME_DIR="$SLIME_DIR" MEGATRON_LM_DIR="$MEGATRON_LM_DIR" \
bash scripts/train_sft.sh
```

对 `demo_only / balanced / correction_heavy` 先做小规模 dev 筛选，test 禁止选配比。Gate：SFT 不劣于 prompt；parser error 降低；switch 样本可重放且错误 action 不入 loss；保留至少 `demo_only` 与 `balanced` checkpoint。

---

## 6. Phase 4：离线 Step-RFT

**目标：** 学习当前公开状态下 action 的格式、工具合法性、参数合法性和局部有效性，先于昂贵在线探索验证局部 credit signal。

Upstream 没有真正的 offline Step-RFT：其 RFT 是按终局 reward/partial score 的 trajectory filtering。因此新增：

```text
tau2_rl_pipeline/verifiers/{format,action,argument,progress}.py
tau2_rl_pipeline/data/{candidate_collector,step_dataset}.py
scripts/03_collect_rollouts.py
scripts/06_train_step_rft.sh
```

对每个 train trajectory prefix，保留原成功 action 并从 SFT checkpoint 采样 K 个候选；只读 verifier 或隔离短 rollout 输出 `format/action/args/progress` breakdown。先用最高 step-score action 的 weighted SFT/rejection-RFT 验证数据链路；再实现 SLIME custom offline objective，对同 prefix K 个 candidate 采用 group-normalized step advantage。它是项目新增实验扩展，不得称为 SLIME 内建功能。

```bash
python scripts/03_collect_rollouts.py --mode prefix_candidates \
  --checkpoint artifacts/checkpoints/sft-balanced --split data/splits/v1/train.jsonl \
  --num-candidates 8 --temperature 0.7 --output data/processed/step_rft_train.jsonl
NUM_GPUS=4 TRAIN_DATA=data/processed/step_rft_train.jsonl \
DEV_DATA=data/processed/step_rft_dev.jsonl INIT_CKPT=artifacts/checkpoints/sft-balanced \
SAVE_DIR=artifacts/checkpoints/step-rft-balanced \
SLIME_DIR="$SLIME_DIR" MEGATRON_LM_DIR="$MEGATRON_LM_DIR" bash scripts/06_train_step_rft.sh
```

报告 format accuracy、valid action accuracy、argument validity、invalid-call rate、错误混淆矩阵以及终局 success；若只提升格式却降低 success，先排查 reward 是否鼓励无意义 read tool。

---

## 7. Phase 5：Terminal-only GRPO Baseline

**目标：** 建立公平的在线 RL 基线，证明后续增益不是“只要上 RL 就会有”。

复制 upstream `scripts/train_grpo.sh` 为 `07_train_terminal_grpo.sh`，复用其 Ray job、SLIME `train.py`、SGLang rollout、Megatron 启动参数与 `tau2_rl_pipeline.rollout.generate`。每 task group 固定 $G=4$ trajectories。reward hook 改为 `reward_terminal.terminal_reward_post_process`，只返回官方 `sample.get_reward_value(args)`。

```bash
TAU2_REWARD_ALPHA=0 TAU2_USE_CURRICULUM=0 TAU2_APPLY_CURRICULUM_WEIGHTS=0 \
NUM_GPUS=4 SFT_CKPT_DIR=artifacts/checkpoints/step-rft-balanced \
REF_CKPT_DIR=artifacts/checkpoints/step-rft-balanced \
SAVE_DIR=artifacts/checkpoints/terminal-grpo TAU2_TRAIN_TASKS_JSONL=data/splits/v1/train.jsonl \
SLIME_DIR="$SLIME_DIR" MEGATRON_LM_DIR="$MEGATRON_LM_DIR" \
bash scripts/07_train_terminal_grpo.sh
```

禁止混入 upstream 默认 partial score、domain-adaptive alpha 或 process-local curriculum。每 50 updates 记录 train/dev success、DB/communication、KL/entropy/loss、reward、rollout length、valid/invalid calls、timeout、policy violation、GPU/env wait。

---

## 8. Phase 6：StateTool-RL（核心方法）

**目标：** 在完全相同的 SLIME rollout/trainer 拓扑上，只替换 reward hook，从而严格归因 state-aware reward 的增益。

定义：

$$
R = R_{terminal} + \alpha\sum_t[\Phi(s_{t+1})-\Phi(s_t)] - \beta C_{tool} - \gamma I_{invalid} - \delta I_{false\_complete} - \eta I_{post\_success\_abort} - \zeta I_{overdue}
$$

在 `tau2_rl_pipeline/reward_state_tool.py` 实现。`Phi` 只能使用公开 state、train/dev 离线 milestone 或 verifier；test 只使用官方终局 evaluator，绝不读取 hidden target/reference milestone。现有 upstream `task_reward + alpha * partial_score` 只由**终局** reward_info 计算，且 DB weight 默认 0；可作为 terminal+partial warm-start 消融，但不能命名为逐状态 reward。

复制 GRPO 脚本为 `08_train_state_tool_grpo.sh`，仅替换 custom reward hook：

```bash
NUM_GPUS=4 SFT_CKPT_DIR=artifacts/checkpoints/step-rft-balanced \
REF_CKPT_DIR=artifacts/checkpoints/step-rft-balanced \
SAVE_DIR=artifacts/checkpoints/state-tool-grpo TAU2_TRAIN_TASKS_JSONL=data/splits/v1/train.jsonl \
SLIME_DIR="$SLIME_DIR" MEGATRON_LM_DIR="$MEGATRON_LM_DIR" \
bash scripts/08_train_state_tool_grpo.sh
```

最少 3 seeds：B2 terminal-only、A3 terminal+progress、A4 full reward、A5 full reward without switch-policy。报告均值、标准差和 paired/bootstrap CI。

---

## 9. Phase 7：Held-out 评测、泛化与失败分析

冻结 checkpoint、prompt、sampling、reward 权重和选择规则后，才运行 `test_id` / `test_ood`：

```bash
python eval/eval_base.py --policy-backend sglang \
  --checkpoint artifacts/checkpoints/state-tool-grpo \
  --task-manifest data/splits/v1/test_id.jsonl --num-samples 4 \
  --output artifacts/evals/final_test_id
```

报告 Task/DB/Communication success、`avg_reward`、any-success `pass^k`、mean/p95 tool calls、invalid-call rate、policy violation rate、p50/p95 episode length/latency、cost per success。每个模型抽样至少 50 失败 episode，分类：意图/澄清、选错工具、参数、observation 推理、policy 违规、不必要副作用、假完成、成功后破坏、timeout、环境/evaluator 异常。

最后在 ToolSandbox 运行冻结超参的跨环境实验；只允许写 parser/logging adapter，不允许再调 reward 权重。

---

## 10. 可执行的细粒度实施清单

下面的任务按依赖顺序执行。每项完成后先跑给出的验证命令，再提交一个小而单一目的的 commit；不要把环境、数据、reward 和训练系统改动混在一个 commit 中。

### P0-A：建立版本与配置记录

**文件：** 创建 `scripts/00_doctor.sh`、`tau2_rl_pipeline/utils/logging.py`、`reports/experiment_registry.csv`。

1. `00_doctor.sh` 输出所有 SHA、`python --version`、`nvidia-smi`、`ray --version`、SGLang 版本和磁盘/共享内存；失败时返回非零。
2. `logging.py` 实现 `write_versions(run_dir, config)`：写入 `versions.json`、resolved config 和环境变量白名单；显式过滤 key/token/header。
3. 验证：`bash scripts/00_doctor.sh && test -s reports/doctor/latest.json`。
4. 提交：`git commit -m "chore: record pinned training environment"`。

### P0-B：实现 policy client 与单任务 runner

**文件：** 修改 `eval/eval_passk.py`；创建 `tau2_rl_pipeline/evaluation/runner.py`、`tests/integration/test_base_runner.py`。

1. 先写 mock policy fixture：第一次返回合法 tool call，第二次返回最终答复；测试 runner 断言 `env.reset → env.step → close` 的顺序和 `terminated=True`。
2. 抽取 upstream attempt loop，定义 `PolicyClient.generate(messages, tools, sampling) -> PolicyResponse`。
3. 实现 `SGLangPolicyClient` 与 `OpenAICompatiblePolicyClient`；provider exception、空响应、HTTP 限流映射为结构化 `PolicyError`。
4. 运行：`pytest tests/integration/test_base_runner.py -v`；预期所有 mock 轨迹都写出终局 status 而非抛异常。
5. 提交：`git commit -m "feat: add provider-agnostic tau2 evaluation runner"`。

### P0-C：持久化轨迹、指标与 smoke 报告

**文件：** 创建 `eval/eval_base.py`、`tau2_rl_pipeline/evaluation/{metrics,failure_analysis}.py`、`data/eval_manifests/airline_smoke_v1.jsonl`。

1. 定义 `TrajectoryStep` / `AttemptResult` dataclass；每次 step 立即 append JSONL，不等 episode 成功才落盘。
2. 聚合器以 JSONL 为唯一输入重算 `avg_reward`、success、DB/communication success、any-success@K、mean/p95 calls 与失败计数。
3. 用 10–20 个显式官方 task ID 填写 smoke manifest；注释其覆盖意图，禁止使用 task 顺序采样。
4. 先执行一个 API task，再执行 K=4 批量 smoke；从 `rollouts.jsonl` 单独运行聚合器并比对 `summary.json`。
5. 验证：`python eval/eval_base.py ...` 后，`python -m tau2_rl_pipeline.evaluation.metrics --rollouts <path>` 的指标逐字段一致。
6. 提交：`git commit -m "feat: add reproducible airline base evaluation"`。

### P1-A：状态审计与 action-error 规范化

**文件：** 修改 `tau2_rl_pipeline/{actions,env}.py`；创建 `tau2_rl_pipeline/utils/state_hash.py`、`tests/envs/test_state_hash.py`。

1. 对合法 action、malformed JSON、未知 tool、schema 不合法参数分别写测试；每种失败必须产生可序列化 error observation 和 error code。
2. 为 snapshot 构造 canonical JSON（稳定 key order、剔除时间戳/随机 id），计算 SHA-256 state hash；测试相同 state 一致、关键 DB 字段不同则不同。
3. 仅在 state 可获得时记录 state hash；不可获得时明确 `state_hash=null` 与原因，禁止伪造。
4. 运行：`pytest tests/envs -v`。
5. 提交：`git commit -m "feat: audit tau2 state and normalize action failures"`。

### P1-B：重放与并发隔离测试

**文件：** 创建 `scripts/01_smoke_tau2.py`、`tests/integration/{test_parallel_reset,test_rollout_replay}.py`。

1. scripted fixture 在固定 task/seed 连续运行三次；比较 initial hash、每步 action/result、final reward/DB digest。
2. 用 process 或 worker pool 并发 8 个 episode；worker A 写状态后，worker B 的 reset hash 仍须不变。
3. 失败时输出最小复现 task ID、seed、worker id 和 state diff；若隔离失败，切换到 process-per-episode 并在配置中将默认并发降为 1。
4. 运行：`pytest tests/integration/test_parallel_reset.py tests/integration/test_rollout_replay.py -v -s`。
5. 提交：`git commit -m "test: enforce replayable isolated tau2 episodes"`。

### P2-A：生成无泄漏 split manifest

**文件：** 修改 `tau2_rl_pipeline/tasks.py`；创建 `tau2_rl_pipeline/data/split.py`、`scripts/02_make_splits.py`、`tests/data/test_split.py`。

1. 从官方 task metadata 抽取 domain/template/entity-pattern；若某字段不存在，显式记录 fallback 规则。
2. 以 group 为原子切分，断言任一 group 不同时出现在 train 与任一 test。
3. 输出各 split JSONL、统计表与 `split_hashes.json`；固定排序保证同 metadata/seed 得到同一文件。
4. 运行：`pytest tests/data/test_split.py -v && python scripts/02_make_splits.py --seed 20260903`。
5. 提交：`git commit -m "feat: create leakage-aware tau2 task splits"`。

### P2-B：冻结 local SGLang prompt baseline

**文件：** 修改 `configs/environment/tau2_airline.yaml`；创建 `configs/training/prompt_baseline.env`。

1. 写入 `MODEL_ID`、SGLang URL、chat-template revision、temperature、max steps、user simulator 模型/URL/temperature。
2. 以该文件启动 SGLang 和 user simulator；不得让训练和评测采用不同 user simulator 默认值。
3. 在 dev manifest 运行两次；把配置 hash、seed、manifest hash 和结果路径写进 registry。
4. Gate：除已声明的模型采样随机性外，运行器和 evaluator 结果一致；否则回到 P1。
5. 提交：`git commit -m "chore: freeze prompting baseline protocol"`。

### P3-A：构造可审计 SFT 数据

**文件：** 创建 `tau2_rl_pipeline/data/{switch_policy,step_dataset}.py`、`scripts/04_build_sft_data.py`、`tests/data/test_loss_mask.py`。

1. 用 fixture 构造 demo、teacher 和 weak-error→teacher-repair 三类样本。
2. 断言 system/user/tool-schema/tool-result/weak-error token labels 全为 ignore index；repair tool call 与最终用户回复 labels 有效。
3. switch-policy 的 teacher prompt 只能来自可见 history；测试中注入 hidden action 时应被 schema 拒绝。
4. 为每条样本保存 `source_task_id`、`trajectory_type`、`handoff_step`、state hash、teacher final status 和数据版本。
5. 运行：`pytest tests/data/test_loss_mask.py -v`，再构建小数据并人工检查两条 JSONL。
6. 提交：`git commit -m "feat: build auditable multi-source sft data"`。

### P3-B：训练与选择 SFT warm start

**文件：** 创建 `configs/data/mixture_v1.yaml`；修改 `scripts/train_sft.sh`。

1. 让脚本显式读取 `NUM_GPUS/HF_DIR/SFT_DATA_JSONL/SAVE_DIR/SLIME_DIR/MEGATRON_LM_DIR/SEED`，启动前检查路径存在。
2. 先以小 token budget 对 `demo_only`、`balanced`、`correction_heavy` 运行 dev；选择规则只能查看 dev success、parser error、invalid-call rate。
3. 使用最佳 1–2 个 mixture 跑完整 SFT；保存 resolved command/config 与 checkpoint manifest。
4. 训练后立即复用 `eval_base.py` 评测 dev，并与 P2 prompt baseline 对比。
5. 提交：`git commit -m "feat: train slime sft warm-start policies"`。

### P4-A：实现并校验 step verifier

**文件：** 创建 `tau2_rl_pipeline/verifiers/{format,action,argument,progress,aggregate}.py`、`tests/verifiers/`。

1. `format` 只判断 parseability、单 tool-call、自然语言 reply 的允许上下文；不能因结果成功倒推格式分。
2. `action` 检查 tool availability 与显式业务 policy；`argument` 检查 schema、类型、枚举和当前公开实体；`progress` 只基于公开状态变化。
3. 每个 verifier 返回 `{score, code, evidence}`，聚合器保留分项，不能只输出总 reward。
4. 为每个错误码写至少一个确定性 fixture；运行 `pytest tests/verifiers -v`。
5. 提交：`git commit -m "feat: add interpretable step verifiers"`。

### P4-B：采 candidate、验证数据链路、训练 Step-RFT

**文件：** 创建 `tau2_rl_pipeline/data/candidate_collector.py`、修改 `scripts/{03_collect_rollouts,06_train_step_rft}.sh`。

1. 对 train prefix 保存 K 个 candidate 的 prompt/state/action/logprob/verifier breakdown；candidate group key 仅由公开 observation/tools/history hash 构成。
2. 首先执行 highest-score candidate 的 weighted SFT/rejection-RFT smoke，确认数据能读取、loss mask 对齐、checkpoint 能加载。
3. 再实现同 prefix 组内 advantage 标准化；方差为零的 candidate group 记录并跳过，不除零、不伪造 advantage。
4. 比较 SFT 与 Step-RFT 的格式、参数、invalid-call 和终局 success；不在 test 上选择 K、温度或权重。
5. 提交：`git commit -m "feat: add offline verifier-guided step rft"`。

### P5-A：建立 terminal-only GRPO 的不可变对照

**文件：** 创建 `tau2_rl_pipeline/reward_terminal.py`、`scripts/07_train_terminal_grpo.sh`、`tests/rewards/test_terminal_reward.py`。

1. 单测断言 reward hook 只返回官方终局 reward；partial score、alpha、curriculum、tool cost、invalid penalty 任一出现即失败。
2. 脚本固定 `TAU2_REWARD_ALPHA=0`、`TAU2_USE_CURRICULUM=0`、`TAU2_APPLY_CURRICULUM_WEIGHTS=0` 并在启动日志打印。
3. SLIME actor/ref、SGLang rollout 与 Ray/Megatron 参数均继承同一个 upstream 骨架；仅 checkpoint/output/reward hook 可不同。
4. 先跑 5–10 updates 的 launch smoke，再运行完整 dev training；监控 KL、entropy、rollout length、timeout、env wait。
5. 提交：`git commit -m "feat: add fair terminal-only slime grpo baseline"`。

### P6-A：实现 StateTool reward 与防泄漏审计

**文件：** 创建 `tau2_rl_pipeline/reward_state_tool.py`、`tests/rewards/test_state_tool_reward.py`。

1. 将 terminal、progress、tool cost、invalid、false-complete、post-success destructive write、overdue 分为独立纯函数，均返回 score/code/evidence。
2. fixture 覆盖：成功、无效参数、重复读工具、假完成、成功后破坏写操作、max-step 截断。
3. reward 函数输入中不得含 test hidden target；写测试验证含 `hidden_reference` 的 payload 被拒绝或忽略。
4. 所有系数从 config 读取；记录 raw component 与 final reward，避免仅靠总分排查 reward hacking。
5. 运行：`pytest tests/rewards/test_state_tool_reward.py -v`。
6. 提交：`git commit -m "feat: add leak-safe state-aware tool reward"`。

### P6-B：运行消融而不改变训练系统

**文件：** 创建 `scripts/08_train_state_tool_grpo.sh`、`configs/training/state_tool_grpo.env`、`scripts/09_reproduce_main_table.sh`。

1. 从 P5 脚本复制；diff 仅允许 reward hook、reward 系数、run name/output dir。用 CI 或 `diff` 检查禁止改变 rollout/trainer/SGLang/Ray 参数。
2. 先在 dev 运行 B2、A1、A2、A3、A4、A5；确定主方法后对 B2/A3/A4/A5 各运行 3 seeds。
3. `09_reproduce_main_table.sh` 读取已完成 run 的 JSONL，而非手工复制数字；输出 mean/std、paired delta、bootstrap CI。
4. 若 KL 爆炸、重复工具调用或 timeout 激增，回滚上一个 stable checkpoint，检查 parser/reward component/env error 后再改一个变量。
5. 提交：`git commit -m "feat: run state-tool reward ablations on slime grpo"`。

### P7-A：冻结最终评测与失败报告

**文件：** 修改 `eval/eval_base.py`；创建 `reports/templates/{main_results,failure_cases}.md`。

1. 将选中的 checkpoint、prompt/sampling、reward 权重、user simulator、split SHA 写为只读 `final_protocol.json`。
2. 对 `test_id`、`test_ood` 运行 K=4；评测脚本拒绝任何 `--tune-*`、train/dev manifest 或 reward milestone 输入。
3. 以每个模型至少 50 个失败 episode 的固定抽样 seed 生成失败分类表，保留每类 2–3 条匿名 trajectory。
4. ToolSandbox 仅加载冻结配置；允许 adapter，不允许更新上述 final protocol。
5. 提交：`git commit -m "docs: publish frozen held-out evaluation report"`。

---

## 11. 测试、Gate 与交付物

单测：parser、format/action/argument verifier、loss mask、state hash、false-complete/post-success/overdue reward fixture。集成测试：single-task smoke、parallel reset、rollout replay、资源清理。

| 阶段 | Gate | 产物 |
|---|---|---|
| P0 | API + official evaluator + airline base 跑通 | JSONL、`avg_reward/pass^k`、reward breakdown、失败表 |
| P1 | reset/replay/并发隔离 | state audit、smoke 日志 |
| P2 | split 冻结、prompt 可复跑 | split SHA、prompt baseline |
| P3 | SFT 不退化、loss mask 正确 | SFT data/checkpoint/mixture 对比 |
| P4 | candidate/verifier 可重放 | Step-RFT checkpoint、错误矩阵 |
| P5 | 纯 terminal GRPO 稳定 | RL 曲线与 B2 |
| P6 | reward 消融完成 | B0–A5 dev 表、reward 分布 |
| P7 | held-out 与泛化结束 | 主表、CI、失败案例、复现脚本 |

每个训练 run 必须输出：`resolved_config.json`、`versions.json`、`split_hashes.json`、`metrics.jsonl`、`checkpoints/`、`rollouts/`、`state_reset_audit.json`。任何 state audit 失败必须非零退出。

---

## 官方入口

- Tau2-RL-Pipeline: https://github.com/jbarnes850/Tau2-RL-Pipeline
- SLIME: https://github.com/THUDM/slime
- SGLang: https://github.com/sgl-project/sglang
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- τ²-Bench: https://github.com/sierra-research/tau2-bench
- τ²-Bench paper: https://arxiv.org/abs/2506.07982
- ToolSandbox paper: https://arxiv.org/abs/2408.04682
