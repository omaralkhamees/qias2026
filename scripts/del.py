"""
Delete specific case IDs from checkpoint.json so they get re-run.

Usage: python delete_from_checkpoint.py <checkpoint_path> <id1> <id2> ...
"""

import json
import sys

if len(sys.argv) < 3:
    print("Usage: python delete_from_checkpoint.py <checkpoint.json> <id1> <id2> ...")
    sys.exit(1)

checkpoint_path = sys.argv[1]
ids_to_delete = set(sys.argv[2:])

with open(checkpoint_path, "r", encoding="utf-8") as f:
    data = json.load(f)

before = len(data)
data = [r for r in data if r["id"] not in ids_to_delete]
after = len(data)

with open(checkpoint_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Removed {before - after} cases. Checkpoint: {before} → {after}")