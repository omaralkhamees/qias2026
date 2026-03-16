from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mawarith_benchmark.evaluation.mire import compute_mire


# -----------------------------
# I/O helpers
# -----------------------------
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(".json root must be a list")
        return data

    if suf == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    raise ValueError(f"Unsupported extension: {path.suffix}")


def parse_json_maybe(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return x
    return x


def extract_ref_output(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # supports row["output"], row["answer"]["output"], or row["answer"] being already output
    out = parse_json_maybe(row.get("output"))
    if isinstance(out, dict):
        return out

    ans = parse_json_maybe(row.get("answer"))
    if isinstance(ans, dict):
        out2 = parse_json_maybe(ans.get("output"))
        if isinstance(out2, dict):
            return out2
        return ans

    return None


def build_ref_index(ref_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for row in ref_rows:
        ex_id = row.get("id")
        if isinstance(ex_id, str):
            out = extract_ref_output(row)
            if isinstance(out, dict):
                idx[ex_id] = out
    return idx


def parse_pred_structured(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = parse_json_maybe(row.get("answer_structured"))
    return raw if isinstance(raw, dict) else None


# -----------------------------
# Eval config
# -----------------------------
@dataclass
class EvalConfig:
    pred_path: Path
    ref_path: Path
    out_path: Optional[Path] = None
    inspect_id: Optional[str] = None


#DEFAULT_EVAL_DIR = Path("mawarith_reasoner/output/evaluation")


def _resolve_output_paths(pred_path: Path, out_path: Optional[Path]) -> Tuple[Path, Path]:
    """
    Evaluation outputs are written NEXT TO the prediction file.

    Example:
      pred_path = .../output/Fanar-Sadiq/pred_fanar.json

    Outputs:
      .../output/Fanar-Sadiq/pred_fanar_eval.csv
      .../output/Fanar-Sadiq/pred_fanar_eval.json
    """
    pred_path = Path(pred_path)
    out_dir = pred_path.parent          # <-- dossier du modèle
    pred_stem = pred_path.stem           # pred_fanar

    csv_path = out_dir / f"eval.csv"
    json_path = out_dir / f"eval.json"

    return csv_path, json_path



# -----------------------------
# Main evaluation
# -----------------------------
def evaluate_predictions(cfg: EvalConfig) -> Dict[str, Any]:
    pred_rows = load_json_or_jsonl(cfg.pred_path)
    ref_rows = load_json_or_jsonl(cfg.ref_path)
    ref_index = build_ref_index(ref_rows)

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    missing_in_ref = 0
    missing_structured = 0

    for row in pred_rows:
        ex_id = row.get("id")
        if not isinstance(ex_id, str):
            continue

        gold = ref_index.get(ex_id)
        if gold is None:
            missing_in_ref += 1
            continue

        pred_struct = parse_pred_structured(row)
        if pred_struct is None:
            missing_structured += 1
            csv_rows.append({
                "id": ex_id,
                "MIR-E": 0.0,
                "heirs_blocked": 0.0,
                "shares": 0.0,
                "awl": 0.0,
                "final": 0.0,
            })
            json_rows.append({
                "id": ex_id,
                "MIR-E": 0.0,
                "details": {"error": "missing_or_bad_answer_structured"},
            })
            continue

        mire = compute_mire(gold, pred_struct)

        csv_rows.append({
            "id": ex_id,
            "MIR-E": mire["MIR-E"],
            "heirs_blocked": mire["subscores"]["heirs_blocked"],
            "shares": mire["subscores"]["shares"],
            "awl": mire["subscores"]["awl"],
            "final": mire["subscores"]["final"],
        })

        json_rows.append({
            "id": ex_id,
            "MIR-E": mire["MIR-E"],
            "subscores": mire["subscores"],
            "details": mire["details"],
        })

    csv_path, json_path = _resolve_output_paths(cfg.pred_path, cfg.out_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "MIR-E", "heirs_blocked", "shares", "awl", "final"])
        w.writeheader()
        w.writerows(csv_rows)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_mire = (sum(r["MIR-E"] for r in csv_rows) / len(csv_rows)) if csv_rows else 0.0
    summary: Dict[str, Any] = {
        "avg_mire": round(avg_mire, 6),
        "missing_in_ref": missing_in_ref,
        "missing_structured": missing_structured,
        "n_scored": len(csv_rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }

    if cfg.inspect_id:
        case = next((r for r in json_rows if r.get("id") == cfg.inspect_id), None)
        if case is not None:
            summary["inspect"] = case

    return summary
