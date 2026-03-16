"""
Evaluation wrapper — bridges to the official QIAS MIR-E evaluator.

Usage:
    from src.evaluation import evaluate_run
    summary = evaluate_run(predictions_path, reference_path, output_dir, evaluator_src)
"""

import json
import sys
from pathlib import Path


def evaluate_run(
    pred_path: Path,
    ref_path: Path,
    output_dir: Path,
    evaluator_src: str,
) -> dict:
    """
    Run the official QIAS evaluator on predictions.

    Args:
        pred_path:     Path to predictions.json
        ref_path:      Path to reference JSON (single file or merged)
        output_dir:    Where to write eval.json
        evaluator_src: Path to qias_shared_task_2026-main/src/

    Returns:
        Summary dict with avg_mire, n_scored, per-case details
    """
    # Add evaluator source to path
    evaluator_src = str(Path(evaluator_src).resolve())
    if evaluator_src not in sys.path:
        sys.path.insert(0, evaluator_src)

    from mawarith_benchmark.evaluation import EvalConfig, evaluate_predictions

    eval_out = output_dir / "eval.json"

    cfg = EvalConfig(
        pred_path=pred_path,
        ref_path=ref_path,
        out_path=eval_out,
    )

    summary = evaluate_predictions(cfg)
    return summary


def merge_reference_files(ref_paths: list[Path], output_path: Path) -> Path:
    """
    Merge multiple reference JSON files into one for the evaluator.

    Args:
        ref_paths:   List of JSON file paths, each containing a list of cases
        output_path: Where to write the merged file

    Returns:
        Path to the merged file
    """
    merged = []
    for p in ref_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Reference file must contain a JSON list: {p}")
        merged.extend(data)

    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def build_analysis(eval_path: Path) -> dict:
    """
    Analyze eval.json to group failures by error pattern.

    Returns:
        Dict with failure_counts, worst_cases, subscore_breakdown
    """
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    perfect = []
    failed = []

    for case in data:
        if case.get("MIR-E", 0) >= 1.0:
            perfect.append(case["id"])
        else:
            failed.append(case)

    # Group failures by which subscore is broken
    subscore_failures = {
        "heirs_blocked": [],
        "shares": [],
        "awl": [],
        "final": [],
    }

    for case in failed:
        subscores = case.get("subscores", {})
        for key in subscore_failures:
            if subscores.get(key, 1.0) < 1.0:
                subscore_failures[key].append({
                    "id": case["id"],
                    "score": subscores[key],
                    "mire": case.get("MIR-E", 0),
                })

    # Sort worst cases
    failed_sorted = sorted(failed, key=lambda x: x.get("MIR-E", 0))

    return {
        "total": len(data),
        "perfect": len(perfect),
        "failed": len(failed),
        "avg_mire": sum(c.get("MIR-E", 0) for c in data) / len(data) if data else 0,
        "worst_10": [
            {"id": c["id"], "mire": c.get("MIR-E", 0)}
            for c in failed_sorted[:10]
        ],
        "subscore_failures": {
            k: len(v) for k, v in subscore_failures.items()
        },
    }
