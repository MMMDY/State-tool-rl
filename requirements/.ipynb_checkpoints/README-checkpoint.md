# 原生环境迁移

这些 lock 文件由 2026-09-03 已验证环境导出，目标为 Linux、Python 3.12 和 CUDA 12.8 的 NVIDIA GPU 主机；不使用 Docker。

## 前置条件

- 安装兼容的 NVIDIA 驱动，并确认 `nvidia-smi` 正常。
- 使用 Python 3.12；系统需要能编译 FlashInfer 的基础工具链。
- 不要在 pip 安装阶段执行 `source /etc/network_turbo`，因为它会切换到本次不可用的阿里源。

## 1. 获取固定源码

```bash
git clone <project-repository> state-tool-rl
cd state-tool-rl
git checkout c0d434b453719e149aac70452325f6e04ac086cf
source /etc/network_turbo  # 仅用于 Git/Hugging Face 下载
git clone https://github.com/THUDM/slime.git third_party/slime
git -C third_party/slime checkout 4c1ab40203952b3dcc8582b653f3a83f2c6e8128
git clone https://github.com/NVIDIA/Megatron-LM.git third_party/Megatron-LM
git -C third_party/Megatron-LM checkout ed6ac270bacb6a4258e1c5b4a911bcf0fd3e4d9b
git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
git -C third_party/tau2-bench checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
```

## 2. 创建两套环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r requirements/sglang-py312-cu128.lock.txt
deactivate

python3.12 -m venv .venv-tau2
source .venv-tau2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/tau2-train-py312-cu128.lock.txt
deactivate
```

`.venv` 只运行 SGLang policy server；`.venv-tau2` 只运行 tau2、Ray、SLIME 和 Megatron。二者因 OpenAI SDK 版本冲突不能合并。

## 3. 验证

```bash
.venv/bin/python -m pip check
.venv-tau2/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

