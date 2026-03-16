"""
Convert predictions from pipeline format to submission format (zipped).

Pipeline format:  {id, source, question, reasoning, answer_structured}
Submission format: {id, question, output}

Usage:
    python convert_predictions.py output/dev/gemini_reason/predictions.json
    python convert_predictions.py output/dev/mistral_small_2506_scrape/predictions.json

Output is saved alongside the original as prediction.zip (containing prediction.json)
"""

import argparse
import json
import zipfile
from pathlib import Path


def convert(input_path: Path) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    converted = [
        {
            "id": entry["id"],
            "question": entry["question"],
            "output": entry["answer_structured"],
        }
        for entry in data
    ]
    json_bytes = json.dumps(converted, ensure_ascii=False, indent=2).encode("utf-8")

    zip_path = input_path.with_name("prediction.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json_bytes)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Convert predictions to submission format")
    parser.add_argument("paths", nargs="+", type=Path, help="Path(s) to predictions.json files")
    args = parser.parse_args()

    # Expand globs (Windows doesn't do it automatically)
    expanded = []
    for p in args.paths:
        matches = sorted(Path(".").glob(str(p)))
        if matches:
            expanded.extend(matches)
        else:
            print(f"SKIP: {p} not found")

    for p in expanded:
        out = convert(p)
        print(f"{p} -> {out}")


if __name__ == "__main__":
    main()
