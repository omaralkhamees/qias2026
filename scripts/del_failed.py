"""
Delete cases from checkpoint.json that scored below 1.0 MIR-E in eval.json.

Usage: python del_failed.py <run_dir>
  - Reads eval.json to find cases with MIR-E < 1.0
  - Removes them from checkpoint.json so they get re-run
"""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python del_failed.py <run_dir>")
    sys.exit(1)

run_dir = Path(sys.argv[1])
eval_path = run_dir / "eval.json"
checkpoint_path = run_dir / "checkpoint.json"

if not eval_path.exists():
    print(f"ERROR: {eval_path} not found")
    sys.exit(1)
if not checkpoint_path.exists():
    print(f"ERROR: {checkpoint_path} not found")
    sys.exit(1)

evals = json.loads(eval_path.read_text(encoding="utf-8"))
failed_ids = {e["id"] for e in evals if e.get("MIR-E", 1.0) < 1.0}

checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
before = len(checkpoint)
checkpoint = [r for r in checkpoint if r["id"] not in failed_ids]
after = len(checkpoint)

checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Removed {before - after} cases (MIR-E < 1.0). Checkpoint: {before} → {after}")