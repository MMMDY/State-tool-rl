#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT_DIR}/reports/doctor"
REPORT_PATH="${REPORT_DIR}/latest.json"
TAU2_PYTHON="${TAU2_PYTHON:-${ROOT_DIR}/.venv-tau2/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

mkdir -p "${REPORT_DIR}"

for required_path in "${TAU2_PYTHON}" "${SGLANG_PYTHON}" \
  "${ROOT_DIR}/third_party/slime" \
  "${ROOT_DIR}/third_party/Megatron-LM" \
  "${ROOT_DIR}/third_party/tau2-bench"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "doctor: missing ${required_path}" >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "doctor: nvidia-smi not found" >&2
  exit 1
fi

"${TAU2_PYTHON}" - "${ROOT_DIR}" "${REPORT_PATH}" "${TAU2_PYTHON}" "${SGLANG_PYTHON}" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root, report_path, tau2_python, sglang_python = map(Path, sys.argv[1:])


def run(args, *, cwd=None, env=None):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_versions(python):
    code = (
        "import importlib.metadata as m, json; "
        "names=['tau2','ray','slime','megatron-core','torch','sglang',"
        "'sgl-kernel','flashinfer-python','litellm','openai']; "
        "installed={d.metadata.get('Name','').lower() for d in m.distributions()}; "
        "print(json.dumps({n: (m.version(n) if n in installed else None) for n in names}))"
    )
    result = run([python, "-c", code])
    if result["returncode"] != 0:
        return {"error": result}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"error": result}


def torch_info(python):
    code = (
        "import json, torch; "
        "print(json.dumps({'version':torch.__version__,"
        "'cuda_version':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'device_count':torch.cuda.device_count(),"
        "'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
    )
    env = os.environ.copy()
    omp = env.get("OMP_NUM_THREADS", "")
    if not omp.isdigit() or int(omp) < 1:
        env["OMP_NUM_THREADS"] = "1"
    result = run([python, "-c", code], env=env)
    if result["returncode"] != 0:
        return {"error": result}
    try:
        return json.loads(result["stdout"].splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": result}


def git_sha(path):
    result = run(["git", "-C", path, "rev-parse", "HEAD"])
    return result["stdout"] if result["returncode"] == 0 else None


def free_space(path):
    result = run(["df", "-P", "-k", path])
    if result["returncode"] != 0:
        return {"error": result}
    lines = result["stdout"].splitlines()
    return lines[-1] if lines else None


def port_status(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        return {"port": port, "listening": sock.connect_ex(("127.0.0.1", port)) == 0}
    finally:
        sock.close()


ports = []
for raw_port in os.environ.get("TAU2_DOCTOR_PORTS", "30000,30001").split(","):
    raw_port = raw_port.strip()
    if raw_port:
        try:
            ports.append(port_status(int(raw_port)))
        except ValueError:
            ports.append({"port": raw_port, "error": "not an integer"})

nvidia = run([
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader",
])
ray_version = run([tau2_python, "-m", "ray", "--version"])

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "project_root": str(root),
    "git": {
        "project": git_sha(root),
        "slime": git_sha(root / "third_party/slime"),
        "megatron": git_sha(root / "third_party/Megatron-LM"),
        "tau2_bench": git_sha(root / "third_party/tau2-bench"),
    },
    "python": {
        "system": platform.python_version(),
        "tau2": run([tau2_python, "--version"]),
        "sglang": run([sglang_python, "--version"]),
    },
    "packages": {
        "tau2_environment": package_versions(tau2_python),
        "sglang_environment": package_versions(sglang_python),
    },
    "torch": {
        "tau2_environment": torch_info(tau2_python),
        "sglang_environment": torch_info(sglang_python),
    },
    "ray": ray_version,
    "gpu": nvidia,
    "storage": {
        "project": free_space(root),
        "shared_memory": free_space("/dev/shm"),
    },
    "ports": ports,
}

required_failures = []
for component, sha in report["git"].items():
    if not sha:
        required_failures.append(f"missing git SHA: {component}")
if nvidia["returncode"] != 0:
    required_failures.append("nvidia-smi failed")
for label, info in report["torch"].items():
    if isinstance(info, dict) and "error" in info:
        required_failures.append(f"Torch check failed: {label}")
if any(item.get("error") for item in ports):
    required_failures.append("invalid doctor port configuration")

report["status"] = "ok" if not required_failures else "failed"
report["failures"] = required_failures
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"report": str(report_path), "status": report["status"], "failures": required_failures}))
if required_failures:
    raise SystemExit(1)
PY
