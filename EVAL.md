# Evaluation baselines

| Baseline | Domain / split | Policy | Policy thinking | Samples per task | Simulator | Simulator thinking | Judge | Policy sampling | Simulator temperature | Judge temperature | Tau2 Pass^1 | Tau2 Pass^4 | Best-of-4 success | Status distribution | Result artifact |
|---|---|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| `retail_test_k4_nothinking_20260913_1920` | retail / test (40 tasks) | `qwen3-4b` via SGLang | off | 4 | `openai/deepseek-flash` | off | `openai/deepseek-flash` | temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_new_tokens=1200 | 0.7 | 0.1 | 0.35625 | 0.075 | 0.675 (27/40) | 152 completed; 8 parse_error; 0 infrastructure_error | `outputs/eval_retail_test_k4_nothinking_20260913_1920/eval_retail_test_k4_nothinking_20260913_1920.json` |
| `airline_test_k4_nothinking_20260913_2045` | airline / test (20 tasks) | `qwen3-4b` via SGLang | off | 4 | `openai/deepseek-flash` | off | `openai/deepseek-flash` | temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_new_tokens=1200 | 0.7 | 0.1 | 0.150 | 0.000 | 0.300 (6/20) | 77 completed; 3 parse_error; 0 infrastructure_error | `outputs/eval_airline_test_k4_nothinking_20260913_2045/eval_airline_test_k4_nothinking_20260913_2045.json` |

Notes:

- `Tau2 Pass^k` is computed with the official `tau2.metrics.agent_metrics.compute_metrics()` implementation: `C(success_count, k) / C(num_trials, k)`. With four rollouts, `Tau2 Pass^4` is non-zero only when all four succeed.
- `Best-of-4 success` means at least one successful rollout among four samples. It is a useful sampling diagnostic, but is not Tau2 `Pass^4`.
