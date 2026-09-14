# Qwen3.5-4B 评测环境恢复记录

更新时间：2026-09-14（重启前）

## 目标

在 `retail` 和 `airline` 的 `test` 集上评测本地
`/root/autodl-tmp/models/Qwen3.5-4B`，每个域均运行两种模式：

- 开启思维链：`--enable-thinking`
- 关闭思维链：不传 `--enable-thinking`

所有评测均使用：SGLang `mem_fraction_static=0.90`、`--num-samples 4`、
`--max-concurrency 32`。完成后将四条结果追加到 `EVAL.md`，不能覆盖已有记录。

## GPU 与网络

- GPU：NVIDIA A800 80GB PCIe；重启前显存空闲。
- 包下载和安装不需要 GPU；仅 SGLang 推理与评测需要 GPU。
- `source /etc/network_turbo` 在此区域只会显示“不支持”，没有可用的网络加速开关。
- pip 使用的镜像：`https://mirrors.aliyun.com/pypi/simple`。

## 已安装的服务环境

虚拟环境：`.venv`（服务使用它；评测器使用 `.venv-tau2`）。重启前确认版本：

| Package | Version |
|---|---:|
| torch | `2.8.0+cu128` |
| sglang | `0.5.10` |
| transformers | `5.3.0` |
| tokenizers | `0.22.1` |
| huggingface-hub | `1.31.0` |
| flashinfer-python | `0.6.7.post2` |
| flashinfer-cubin | `0.6.7.post2` |
| apache-tvm-ffi | `0.1.11` |
| gguf | `0.19.0` |
| sglang-kernel | `0.4.1` |

已安装的 `sglang-kernel 0.4.1` 与当前 `torch 2.8.0` 二进制不兼容：加载
`flash_ops.abi3.so` 会报缺失符号 `c10::SymInt::maybe_as_int_slow_path`，Qwen3.5
模块因此无法加载。必须升级服务环境到 `torch==2.9.1`（以及 `torchaudio==2.9.1`）。

## 重启后恢复步骤

在仓库根目录执行。先确认没有残留服务：

```bash
ps -eo pid,ppid,stat,etime,cmd | rg '[s]glang.launch_server|[e]val_passk' || true
```

安装与当前 SGLang kernel 二进制匹配的 PyTorch。此前下载到 595.5/899.7 MB 时被主动取消；
临时文件不可靠，需重新执行：

```bash
.venv/bin/python -m pip install --no-deps 'torch==2.9.1' 'torchaudio==2.9.1'
```

验证关键版本与二进制导入：

```bash
.venv/bin/python - <<'PY'
import torch, sgl_kernel
from sgl_kernel.flash_attn import flash_attn_varlen_func
print(torch.__version__)
print('sgl_kernel flash attention: OK')
PY
```

启动服务（关闭旧 `sitecustomize` 钩子；SGLang 0.5.10 已改变其 CustomOp API）：

```bash
MODEL_DIR=/root/autodl-tmp/models/Qwen3.5-4B \
GPUS=0 TP=1 MEM_FRACTION=0.9 \
SGLANG_MODEL_NAME=qwen3.5-4b \
SGLANG_FORCE_NATIVE_CUDA_OPS=0 \
bash scripts/start_policy.sh
```

服务日志必须出现 `mem_fraction_static=0.9` 及监听 `127.0.0.1:30000` 的就绪信息。

## 四次评测命令

开启思维链的默认采样配置来自 `configs/qwen3-4b.yaml`：temperature=0.6、top_p=0.95、
top_k=20、max_tokens=2048。关闭思维链时沿用项目对应 no-thinking 默认配置。

```bash
# retail, thinking on
.venv-tau2/bin/python eval/eval_passk.py --sglang-url http://127.0.0.1:30000 --sglang-model qwen3.5-4b --domains retail --task-split test --num-samples 4 --max-concurrency 32 --enable-thinking --output outputs/eval_retail_test_k4_qwen35_thinking_mem090.json

# retail, thinking off
.venv-tau2/bin/python eval/eval_passk.py --sglang-url http://127.0.0.1:30000 --sglang-model qwen3.5-4b --domains retail --task-split test --num-samples 4 --max-concurrency 32 --output outputs/eval_retail_test_k4_qwen35_nothinking_mem090.json

# airline, thinking on
.venv-tau2/bin/python eval/eval_passk.py --sglang-url http://127.0.0.1:30000 --sglang-model qwen3.5-4b --domains airline --task-split test --num-samples 4 --max-concurrency 32 --enable-thinking --output outputs/eval_airline_test_k4_qwen35_thinking_mem090.json

# airline, thinking off
.venv-tau2/bin/python eval/eval_passk.py --sglang-url http://127.0.0.1:30000 --sglang-model qwen3.5-4b --domains airline --task-split test --num-samples 4 --max-concurrency 32 --output outputs/eval_airline_test_k4_qwen35_nothinking_mem090.json
```

输出报告实际位于 `outputs/<run-name>/<run-name>.json`。记录每次报告中的 Pass^1、Pass^4、
Best-of-4、状态分布和采样配置；在 `EVAL.md` 中明确注明 `mem_fraction_static=0.90`。

## 工作区改动提醒

- `EVAL.md` 已有用户此前的 Qwen3-4B 结果，勿覆盖。
- `scripts/sitecustomize.py` 被临时改为从 `sglang.srt.utils.custom_op` 导入；新版 SGLang
  下不应启用它，所以服务命令固定使用 `SGLANG_FORCE_NATIVE_CUDA_OPS=0`。
- 尚未产生任何 Qwen3.5-4B 的有效评测输出，也尚未向 `EVAL.md` 写入 Qwen3.5 结果。
