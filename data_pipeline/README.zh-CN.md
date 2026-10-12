# Qwen3-4B RFT 数据流水线

本目录实现 `docs/plan/2026-09-16_rft_data_pipeline_qwen3-4b.md` 中的必选流程：

1. 仅从 tau2 `train` split 采样 Qwen3-4B 多轮工具调用轨迹；
2. 对每条轨迹执行 schema、process、observation/state replay 和 terminal 三层验证；
3. 规范化、去重、难度分层后导出 SFT JSONL、recovery 候选和质量报告。

正式数据不会随代码或测试自动生成。以下可选项目前不实施：LLM-as-Judge、长思维链改写、历史通过率预筛和 `latest` 软链接。默认关闭 thinking。

## 环境要求

- 使用 `.venv-tau2` 执行采样、验证和导出；
- 在 `model.api_base` 启动 Qwen3-4B 的 SGLang 服务；
- 使用本地用户模拟器时设置 `TAU2_USER_API_BASE`；
- 保持 `third_party/tau2-bench` 子模块可用。

默认配置见 [`configs/rft_qwen3_4b.yaml`](configs/rft_qwen3_4b.yaml)。采样时会把解析后的配置和任务快照写入 run 目录；复用已有 `run_id` 时，如果配置或任务快照发生变化会直接报错，避免混合不同实验。

## 运行顺序

从仓库根目录执行：

```bash
.venv-tau2/bin/python data_pipeline/scripts/sample_rft.py \
  --config data_pipeline/configs/rft_qwen3_4b.yaml \
  --run-id 20260917_1200_qwen3-4b_checkpoint_gitsha

.venv-tau2/bin/python data_pipeline/scripts/verify_rft.py \
  --input data_pipeline/RFT_data/runs/RUN_ID/rollouts/raw.jsonl \
  --output data_pipeline/RFT_data/runs/RUN_ID/rollouts/verified.jsonl

.venv-tau2/bin/python data_pipeline/scripts/export_rft.py \
  --verified data_pipeline/RFT_data/runs/RUN_ID/rollouts/verified.jsonl \
  --output data_pipeline/RFT_data/runs/RUN_ID/exports/rft_train.jsonl
```

也可以在 editable 安装后使用 `sample_rft`、`verify_rft` 和 `export_rft` 三个 console command。三阶段均可独立重跑；JSONL 是 append-only，单条 rollout 完成后立即 fsync，已存在的 trajectory ID 会跳过。阶段 metadata 记录输入和配置 hash，报告与 manifest 使用原子写入。

## 采样策略

每个任务按 `16 → 24 → 32` 个候选自适应扩展，得到 4 条不同且临时合格的动作序列后停止；临时合格只用于节省采样成本，最终结果以 `verify_rft` 为准。成功和失败候选都会保留。

任务级随机种子为：

```text
base_seed + sha256(domain/task_id) + rollout_index
```

raw 记录包含模型/checkpoint、采样参数、task/environment seed、任务初始快照 hash、system prompt/tool schema hash、完整 completion、解析后的动作、逐步 observation、前后 state hash、`SimulationRun` 和终态结果。采样只读取 train；dev/test 仅用于 task ID 泄漏检查，tau2 的 `base` 不视为 held-out split。

## 三层硬验证

轨迹只有通过全部检查才会进入 `verified.jsonl`：

- schema：工具名已声明，参数 JSON 可解析，类型、必填字段、枚举和额外字段均合法；
- process：state hash 链连续，环境拒绝动作、重复无效动作、无进展、过早完成和成功后继续破坏状态都会被拒绝；
- terminal/replay：在新 tau2 环境中重放每个 observation，检查 task ID、初始快照、当前工具 schema、逐步/终态 state hash，并要求官方 `evaluate_simulation(..., strict_replay=True)` 成功。

失败轨迹不会丢弃，而是写入 `rollouts/rejected.jsonl`，并保留可解释的 flags，例如 `invalid_schema`、`bad_args`、`invalid_precondition`、`observation_mismatch`、`replay_mismatch`、`terminal_fail` 和 `overlong`。

## 导出与质量报告

`export_rft` 先按 action fingerprint 去重，再按 state + semantic fingerprint 二次去重；简单任务最多保留 2 条，中等/困难任务最多 4 条。难度按前 16 条候选的 verified 通过率划分：`<20%` 为 hard，`20%~80%` 为 medium，`>80%` 为 easy。配置中的 30/50/20 只是目标比例，不会为了凑比例制造样本。

导出文件包含：

- `exports/rft_train.jsonl`：现有 SFT 流程可消费的 `prompt`/`messages`，保留 system/user/assistant/tool 顺序、工具 schema/hash、质量和 provenance 字段；
- `exports/rft_recovery_candidates.jsonl`：困难任务失败前缀、错误动作和 observation，固定 `training_eligible=false`，不混入正样本；
- `reports/summary.json`：阶段计数、任务难度、拒绝原因、长度/工具调用分位数、replay/重复率、隐私扫描和告警；
- `reports/sample_review.jsonl`：按 domain 的确定性抽检样本和拒绝 flag 覆盖；
- `manifest.json`：代码、模型、tokenizer、配置、环境版本及生成文件 SHA-256。

训练时可将 `SFT_DATA_JSONL` 指向 `exports/rft_train.jsonl`。

## 小范围试跑

建议先复制配置到 `/tmp`，将 `domains` 限为一个 domain、`max_tasks_per_domain=1`、候选数设为 1，并把 `run.root` 指向 `/tmp`。这样可以验证 policy 服务、用户模拟器、tool parser、state replay、terminal verifier 和导出链路，而不会写入 `data_pipeline/RFT_data`。模型未完成任务时也应生成 rejected、recovery、summary 和 manifest，这属于有效的流程 smoke test。

