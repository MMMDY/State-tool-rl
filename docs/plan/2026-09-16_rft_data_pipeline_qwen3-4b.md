# Qwen3-4B RFT 数据制造与筛选方案

## 1. 目标与适用范围

本方案用于按照 [`other_CV_reference_technique.md`](other_CV_reference_technique.md) 制造第一版 RFT（Rejection-sampling Fine-Tuning）数据：使用 Qwen3-4B 进行多候选轨迹采样，只保留“任务终态成功 + 过程可验证”的高质量轨迹，最终转换为 SFT 可直接消费的 JSONL。数据用于在线 GRPO/DAPO 前的轻量冷启动，不替代后续 recovery、DPO 或在线 RL 数据。

范围限定如下：

- 候选任务只来自 `train` split；`dev/test` 永不进入采样、筛选或训练回灌。
- 主采样模型为 Qwen3-4B（记录具体 checkpoint）；环境、工具 schema、system prompt 必须版本化。
- 每个任务保留成功和失败轨迹的统计信息，但训练集默认只写入通过筛选的正样本。
- 所有改写、裁剪、格式修复后的轨迹必须重新执行 schema verifier、state replay 和 terminal verifier。

## 2. 端到端流程

```text
train tasks
   │
   ├─ 离线难度预筛（可选：历史通过率 20%~80%）
   │
   ├─ Qwen3-4B rollout（16 → 24 → 32，达到配额即停）
   │
   ├─ schema verifier + process verifier + terminal verifier
   │
   ├─ 轨迹规范化、状态重放、动作序列去重
   │
   ├─ 可选 LLM-as-Judge 多维 yes/no 二次筛选（默认关闭，不取代规则 verifier）
   │
   ├─ 可选长思维链改写 → 全量复验
   │
   ├─ 人工/脚本抽检、泄漏检查、任务难度分层
   │
   └─ 导出 RFT SFT JSONL + manifest + 质量报告
```

## 3. 目录与产物约定

在已创建的 `data_pipeline/` 下采用以下结构；运行目录名使用时间戳或 git commit，禁止覆盖已有运行：

```text
data_pipeline/
├── configs/
│   └── rft_qwen3_4b.yaml
├── prompts/
│   ├── system_qwen3_tool_v{schema}.txt
│   └── judge_rft_v1.txt                 # 可选 Judge prompt
├── scripts/                         # 后续实现 sampling/filter/export CLI
├── RFT_data/
│   ├── runs/{run_id}/
│   │   ├── config.resolved.yaml
│   │   ├── tasks.snapshot.jsonl
│   │   ├── rollouts/raw.jsonl       # 全部候选（含失败）
│   │   ├── rollouts/verified.jsonl
│   │   ├── rollouts/rejected.jsonl
│   │   ├── judge/scores.jsonl          # 仅启用 Judge 时产生
│   │   ├── rewritten/verified.jsonl
│   │   ├── exports/rft_train.jsonl
│   │   ├── exports/rft_recovery_candidates.jsonl
│   │   ├── reports/summary.json
│   │   └── reports/sample_review.jsonl
│   └── latest -> runs/{run_id}          # 可选软链接
└── README.md
```

建议的 `run_id`：`YYYYMMDD_HHMM_qwen3-4b_<checkpoint>_<gitsha>`。manifest 至少记录模型路径/hash、tokenizer、采样参数、任务文件 hash、环境/工具版本、代码 commit、随机种子和运行时间。

## 4. 采样配置

### 4.1 默认参数（第一轮）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| 初始候选数 | 16/task | 每任务第一批 rollout |
| 追加候选数 | 8/task | 未达配额时追加一次 |
| 单任务上限 | 32/task | 超过即停止 |
| 成功配额 | 4/task | 双重 verifier 通过且去重后的轨迹 |
| temperature | 0.7 | 与现有 Qwen3-4B rollout 配置对齐 |
| top_p | 0.9 | 必须写入 manifest |
| max response tokens | 4096 | 超限标记 `overlong`，不静默截断 |
| thinking | 关闭 | 默认不请求或保留思维链；显式开启时才保留 reasoning 字段并记录配置 |
| seed | 任务级确定性 seed | `base_seed + hash(task_id)`，并记录 rollout index |

采样采用任务自适应停止：16 条后若已得到 4 条合格且去重成功轨迹则停止；否则追加 8 条，最多 32 条。简单任务（前 16 条合格数 ≥13）只保留最多 2 条代表轨迹，避免模板主导；困难任务（通过率 <20%）的合格轨迹进入 recovery 候选池并单独统计，不因配额不足而伪造正样本。

### 4.2 任务快照与可复现

每次 rollout 必须保存任务原文、task_id、domain、split、环境 seed、初始 state snapshot、用户 persona、system prompt hash、tool schema hash。每一步保存：`step_idx`、assistant raw text、解析后的 tool call、tool observation、state hash 前后值、step reward/flags、终止原因。禁止只保存最终文本而丢失状态。

## 5. 轨迹数据 schema

### 5.1 原始 rollout（`raw.jsonl`）

每行一个完整 episode，推荐字段：

```json
{
  "run_id": "...",
  "trajectory_id": "task123_seed7_r03",
  "task_id": "...",
  "domain": "retail",
  "split": "train",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "checkpoint": "...",
  "sampling": {"temperature": 0.7, "top_p": 0.9, "seed": 123, "rollout_index": 3},
  "env": {"version": "...", "seed": 7, "initial_state_hash": "..."},
  "messages": [],
  "steps": [],
  "terminal": {"done": true, "success": false, "official_score": 0.0, "reason": "..."},
  "verifier": {"schema": {}, "process": {}, "terminal": {}},
  "status": "raw"
}
```

### 5.2 导出 RFT SFT 样本（`exports/rft_train.jsonl`）

每行一个可训练样本，至少包含 `task_id`、`trajectory_id`、`messages`、`tools/schema_hash`、`quality`、`source_run_id`。`messages` 使用项目现有 SFT 格式（system/user/assistant/tool），保留合法 tool call 与 observation 的顺序；若训练框架支持 loss mask，则只对 assistant 目标 token 计算 loss，并按实验决定是否 mask `<think>`。

同时导出 `rft_recovery_candidates.jsonl`：包含困难任务的失败前缀、错误动作、错误 observation 和后续可恢复上下文，供 switch-policy recovery 数据制造，不直接混入 RFT 正样本。

## 6. 三层验证与硬门槛

任何轨迹只有在三层检查全部通过后才可进入 `verified.jsonl`：

1. **Schema verifier**：JSON/tool-call 可解析；工具名存在；参数类型、必填字段、枚举值和权限合法；无额外伪造字段。
2. **Process verifier**：每步动作与当前状态 precondition 相容；读取并利用最近 observation；状态 hash 或可计算 partial score 有真实推进；不重复已证明无效的调用；无越权、编造结果、过早宣称完成、成功后继续破坏状态等行为。
3. **Terminal verifier**：重新在隔离环境执行轨迹，终态满足任务 assertion/官方 success；不能仅凭模型最终文本判定成功。

硬拒绝条件包括：终态失败、任一非法 tool call、state replay 不一致、观察值与环境不一致、编造完成、关键步骤缺失、超长导致截断、跨 split 或任务状态泄漏。

建议输出可解释的 verifier flags，例如 `invalid_schema`、`wrong_tool`、`bad_args`、`ignored_observation`、`repeated_invalid_action`、`no_progress`、`false_complete`、`terminal_fail`、`replay_mismatch`。

## 7. 规范化、去重与难度分层

先对 assistant 文本、tool 参数排序、空白和 trace id 做规范化，再计算三类 fingerprint：

- `action_fingerprint`：工具名 + 规范化参数序列；
- `state_fingerprint`：关键状态 hash 序列；
- `semantic_fingerprint`：去除措辞后对诊断/动作意图的稳定表示。

同一任务优先按 `action_fingerprint` 去重；动作不同但状态和意图完全相同的样本只保留 verifier 分数更高、长度更短者。统计每题 `n_rollout`、`n_terminal_success`、`n_process_pass`、`n_both_pass`、`success_rate`。按前 16 条的双重通过率分层：困难 `<20%`、中等 `20%~80%`、简单 `>80%`；边界任务保留原始比例，不强行移动。

训练配比初始建议：困难 30%、中等 50%、简单 20%；每题设置上限，简单题最多 2 条，困难题可保留 4 条。配比作为可配置项并在报告中记录实际值。

## 8. LLM-as-Judge 二次筛选（可选）

Judge 默认关闭，规则 verifier 通过的轨迹可不经 Judge 直接进入后续质检与导出。启用时，Judge 不能推翻 terminal/process verifier 的硬结论，只负责识别“答案对但过程质量差”。使用固定 judge prompt 和温度 0，要求每个维度输出 `yes/no + 简短依据`，输入中隐藏模型名与候选排序。建议维度：

- `correct_diagnosis`：是否准确识别用户目标和当前状态；
- `observation_grounded`：是否引用真实 observation 而非臆测；
- `action_necessity`：动作是否必要且参数最小充分；
- `recovery_quality`：曾失败时是否采取可恢复、非重复的动作；
- `no_hallucination`：是否无编造工具结果/完成状态；
- `concise_reasoning`：是否存在明显凑字数、循环或无关思维链。

启用 Judge 时的接受条件：前五项全为 yes，且 `concise_reasoning=yes`；judge 输出不可解析、理由与轨迹矛盾或置信度不足时进入人工抽检，不自动入训。保存 judge prompt/version、原始响应和解析结果，避免只保留总分。

## 9. 长思维链改写（可选）

本步骤仅适用于显式开启 thinking 的实验。仅对规则验证通过，且经 Judge 或人工标记为 `concise_reasoning=no` 的样本进行改写。改写器只能压缩解释，不能改变 tool call、参数、动作顺序和 observation；改写前后必须保持同一 `action_fingerprint`。改写后的轨迹重新执行三层 verifier，并与原轨迹做 state-by-state diff。任一差异、终态变化或 schema 错误即丢弃改写版，保留原版供人工复核。

## 10. 质检与防数据污染

每个 run 生成 `reports/summary.json`，包含各阶段计数、按 domain/task 难度分布、拒绝原因 Top-K、平均/分位轨迹长度、tool 调用数、终态成功率和去重率。抽检建议：每个 domain 至少 50 条，且覆盖简单/中等/困难、所有主要拒绝 flag；人工复核与自动 verifier 结果计算一致率。

必须执行以下检查：

- train/dev/test task_id 不重叠，且导出文件只含 train；
- 轨迹中的答案、工具结果、数据库状态均来自对应 seed 的 replay；
- 近重复检测（文本 n-gram 或 embedding）防止同一模板占比过高；
- 检查个人信息、敏感字段和内部路径，按项目规范脱敏；
- manifest、配置、代码 commit 与数据文件 hash 可追溯。

建议验收阈值（第一版可作为告警而非硬拒绝）：规则 verifier 通过轨迹人工一致率 ≥98%，导出样本 replay 成功率 100%，重复率 <20%，简单任务样本占比 ≤25%。仅在启用 Judge 时统计 judge 拒绝率，建议告警区间为 10%~50%。

## 11. 实现接口与运行顺序

后续在 `data_pipeline/scripts/` 实现三个必需命令和一个可选 Judge 命令，均可独立重跑：

```text
sample_rft --config configs/rft_qwen3_4b.yaml --run-id ...
verify_rft --input rollouts/raw.jsonl --output rollouts/verified.jsonl
judge_rft  --input rollouts/verified.jsonl --output judge/scores.jsonl
export_rft --verified ... [--judge ...] --output exports/rft_train.jsonl
```

`judge_rft` 为可选步骤；未启用时，`export_rft` 只依赖规则 verifier 结果及其他必需质检。每个阶段采用 append-only 输出和断点续跑；输入文件 hash 不变时可复用缓存。推荐执行顺序：先用 20~50 个 train task 做 dry-run，确认 tool parser、state replay、终态 verifier 和 SFT tokenizer；再扩展全量任务。出现 replay mismatch、全组 reward 为 0、单一 action 模板异常集中或显存/吞吐异常时暂停扩展并保留 raw 数据。

## 12. 实验矩阵与里程碑

第一阶段固定默认采样参数（包括 `thinking=off`），比较仅规则 verifier 与规则 + Judge 两组导出数据。若需评估长思维链改写，另建显式设置 `thinking=on` 的可选实验，不与默认关闭 thinking 的结果混合。第二阶段比较困难/中等/简单配比（30/50/20 与 20/60/20），保持总 token 预算一致。第三阶段训练 RFT checkpoint，与 base Qwen3-4B 在 tau2 train 之外的固定 dev/test 上比较：任务成功率、格式错误率、平均 tool 调用数、平均输出 token、重复无效调用率和 replay 一致率。

里程碑：

1. dry-run 通过：≥20 个任务、每题 16 条 rollout、所有字段和 replay 可复现；
2. pilot：≥200 个 train task，完成抽检和阈值校准；
3. full run：冻结配置与 prompt，生成带 manifest/hash 的正式 `rft_train.jsonl`；
4. RFT 训练后复评，确认没有 dev/test 泄漏及 tool-call 格式退化，再进入 GRPO/DAPO。

## 13. 风险与回滚

- **模型只学模板**：提高 semantic 去重、限制简单题配额，增加困难 recovery 候选。
- **终态成功但过程投机**：提高 process verifier 权重，发现 false-complete 直接硬拒绝。
- **judge 偏置（仅启用时）**：固定 prompt/温度，保存 rationale，定期用人工金标准校准，不让 judge 单独决定入训。
- **状态不可重放**：停止导出，修复 snapshot/replay；不可修复的轨迹只进入 badcase 池。
- **数据规模或显存超限**：按 domain 分片运行，保持每片 manifest，最后用 hash 校验合并，绝不删除 raw 候选。

正式发布前应冻结 `rft_qwen3_4b.yaml`、verifier 版本和任务 snapshot；启用 Judge 时还应冻结 judge prompt。将 `summary.json`、抽检记录与导出文件一起归档。
