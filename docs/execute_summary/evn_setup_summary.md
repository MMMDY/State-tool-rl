# 环境部署总结（原生 Python/CUDA，非 Docker）

日期：2026-09-03
项目提交：`c0d434b453719e149aac70452325f6e04ac086cf`

## 结论

第一阶段基础环境已部署完成，未安装或使用 Docker。推理服务和训练/评测因依赖冲突，按进程边界拆为两个虚拟环境；均已通过 `pip check` 与核心导入验证。

## 主机条件

- Python：3.12.3
- GPU：NVIDIA GeForce RTX 5090，32607 MiB；驱动 595.71.05
- CUDA PyTorch：`2.8.0+cu128`，`torch.cuda.is_available()` 已验证为 `True`
- `/dev/shm`：45 GB；venv 通过 `--system-site-packages` 复用主机 CUDA PyTorch。

## 上游代码（本地克隆，未纳入版本控制）

| 组件 | 路径 | 固定提交 |
| --- | --- | --- |
| SLIME | `third_party/slime` | `4c1ab40203952b3dcc8582b653f3a83f2c6e8128` |
| Megatron-LM | `third_party/Megatron-LM` | `ed6ac270bacb6a4258e1c5b4a911bcf0fd3e4d9b` |
| tau2-bench | `third_party/tau2-bench` | `a2c024725189473d2d7cea3a5cfdbcc67478e41f` |

`pyproject.toml` 已加入 `allow-direct-references = true`，以允许 Hatch 处理项目声明的 Git 直接依赖。

## 已安装环境

### `.venv`：SGLang policy server

- `sglang==0.5.1`
- `sglang-router==0.3.2`
- `sgl-kernel==0.3.5`
- `flashinfer-python==0.2.11.post3`
- 主机 `torch==2.8.0+cu128`
- 已验证 SGLang、router、CUDA 导入和 `python -m sglang.launch_server --help`；`pip check` 通过。

```bash
source .venv/bin/activate
python -m sglang.launch_server --help
```

### `.venv-tau2`：评测、SLIME/Ray 与 Megatron 训练端

- `tau2==1.0.1`
- `ray==2.58.0`
- `slime==0.3.2`（本地 editable）
- `megatron-core==0.20.0+ed6ac270b`（本地 editable）
- 主机 `torch==2.8.0+cu128`
- 已验证 `tau2`、`ray`、`slime`、`megatron.core` 导入；`pip check` 通过。

```bash
source .venv-tau2/bin/activate
```

## 关键兼容性决策

`tau2==1.0.1` 经 LiteLLM 需要 `openai>=2.8.0`；与 CUDA PyTorch 2.8 可用的 `sglang==0.5.1` 需要 `openai==1.99.1`。二者不能共存，故以 SGLang HTTP 服务和 tau2/训练进程为边界拆分 venv。未采用 `sglang==0.5.18`，因为它会固定 `torch==2.13.0`。FlashInfer 用 `--no-build-isolation` 安装，避免构建隔离环境重复下载大型 Torch 依赖。

## 下载网络记录

- Git 与 Hugging Face 下载可先执行 `source /etc/network_turbo` 加速。
- 本机该脚本会将 pip 指向阿里 PyPI；本次阿里源无法解析/提供 `hatchling`、`httpx`、`sglang`，且缺少 `sgl-kernel==0.3.5`，不适合作为本项目 pip 源。
- 已验证可用 pip 源：`https://mirrors.cloud.tencent.com/pypi/simple`。
- 后续 pip 安装显式使用腾讯源；若 CUDA/厂商包未镜像，再临时回退官方源。

## 迁移导出

- SGLang policy server：`requirements/sglang-py312-cu128.lock.txt`
- tau2 评测与训练端：`requirements/tau2-train-py312-cu128.lock.txt`
- 新机原生部署步骤：`requirements/README.md`
- 两份 lock 文件固定完整 Python 包版本、cu128 PyTorch wheel 与本地 editable 源码相对路径；仍须按本文记录先克隆 `third_party/` 的固定提交。

## 当前限制与下一步

- 尚未下载模型、设置模型/API key 或启动实际 SGLang 服务。
- Megatron 会提示未安装 Transformer Engine（TE）和 Apex，并回退 PyTorch 实现；不阻塞 P0 API 评测，但 P3 SFT、P5/P6 RL 前须单独评估/安装。
- 当前只有一张 32 GB GPU；上游默认的 4 GPU、TP=2 参数不能直接使用。训练前须单卡参数化并完成 smoke test。
- 后续进入 P0-A：实现 `scripts/00_doctor.sh`、日志工具和实验登记表，再开始评测。
