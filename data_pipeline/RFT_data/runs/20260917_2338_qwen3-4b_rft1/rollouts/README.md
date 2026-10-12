# Rollout files

The raw rollout JSONL is split into numbered parts so each file stays below
GitHub's 100 MB per-file limit. The parts are ordered and can be reconstructed
with:

```bash
cat raw.part-*.jsonl > raw.jsonl
```

Each part ends at a complete JSONL record boundary.
