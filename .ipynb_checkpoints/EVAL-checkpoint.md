# Evaluation baselines

| Baseline | Domain / split | Policy | Policy thinking | Samples per task | Simulator | Simulator thinking | Judge | Policy sampling | Simulator temperature | Judge temperature | Pass@1 | Pass@4 | Status distribution | Result artifact |
|---|---|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|---|
| `retail_test_k4_nothinking_20260913_1920` | retail / test (40 tasks) | `qwen3-4b` via SGLang | off | 4 | `openai/deepseek-flash` | off | `openai/deepseek-flash` | temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_new_tokens=1200 | 0.7 | 0.1 | 0.375 (15/40) | 0.650 (26/40) | 152 completed; 8 parse_error; 0 infrastructure_error | `outputs/eval_retail_test_k4_nothinking_20260913_1920/eval_retail_test_k4_nothinking_20260913_1920.json` |

Notes:

- `Pass@4` means at least one successful rollout among the four samples; it is not the unbiased Pass@k estimator used by some leaderboards.
