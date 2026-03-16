"""
QIAS 2026 — Al-Mawarith Pipeline

Single entry point for:
  - Sampling cases from train set
  - Running predictions (with checkpointing)
  - Evaluating with official MIR-E scorer
  - Analyzing results

Usage:
    python run.py                          # run with config.yaml defaults
    python run.py --name test_run          # save output to output/test_run/
    python run.py --retry-failed           # re-run only failed cases
    python run.py --analyze-only           # skip prediction, evaluate + analyze
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.providers.base import get_provider
from src.parsing import extract_reasoning, extract_json
from src.evaluation import evaluate_run, build_analysis


# ── Dataset loading ───────────────────────────────────────────

def load_all_cases(dataset_dir: Path, subset: str) -> list[dict]:
    if subset == "dev":
        search_dir = dataset_dir / "dev"
    elif subset == "train":
        search_dir = dataset_dir / "train" / "2026"
    elif subset == "test":
        search_dir = dataset_dir / "test"
    else:
        raise ValueError(f"Unknown subset: {subset}. Use 'train', 'dev', or 'test'")

    files = sorted(search_dir.glob("*.json"))
    all_cases = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            all_cases.extend(data)
    return all_cases


def sample_cases(cases: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(cases):
        print(f"[WARN] Requested {n} samples but only {len(cases)} available. Using all.")
        return cases
    return random.Random(seed).sample(cases, n)


# ── Prediction ────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in data}


def save_checkpoint(path: Path, completed: dict):
    path.write_text(
        json.dumps(list(completed.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_predictions(
    provider,
    prompt: str,
    cases: list[dict],
    output_dir: Path,
    method: str = "llm_reasoning",
    retry_failed: bool = False,
) -> Path:
    checkpoint_path = output_dir / "checkpoint.json"
    predictions_path = output_dir / "predictions.json"
    checkpoint = load_checkpoint(checkpoint_path)

    if retry_failed:
        failed_ids = [k for k, v in checkpoint.items() if "error" in v]
        for fid in failed_ids:
            del checkpoint[fid]
        print(f"Retrying {len(failed_ids)} failed case(s).")
        save_checkpoint(checkpoint_path, checkpoint)

    cached = sum(1 for v in checkpoint.values() if "error" not in v)
    if cached:
        print(f"Resuming — {cached} case(s) already done.")

    total = len(cases)
    errors = 0

    # Initialize browser globally if scraping method is used
    browser = None
    playwright_context = None
    if method == "llm_scraping":
        from src.scraper import solve_with_scraper
        from playwright.sync_api import sync_playwright
        playwright_context = sync_playwright().start()
        browser = playwright_context.chromium.launch(headless=True)
        print("  [Scraper] Chromium browser started.")

    try:
        for i, case in enumerate(cases):
            case_id = case["id"]
            question = case["question"]

            if case_id in checkpoint and "error" not in checkpoint[case_id]:
                continue

            label = f"{i+1}/{total}"
            print(f"  [{label}] {case_id} ...", end=" ", flush=True)
            raw = None

            try:
                # 1. LLM Step
                raw = provider.call(prompt, question, label=label)
                reasoning = ""
                answer = None
                
                
                if method == "llm_reasoning":
                    reasoning = extract_reasoning(raw or "")
                    answer = extract_json(raw)
                    if answer is None:
                        raise ValueError("JSON parse failed")
                        
                elif method == "llm_scraping":
                    parsed_nlp = extract_json(raw)
                    if not parsed_nlp or "heirs" not in parsed_nlp:
                        raise ValueError(f"Failed to parse LLM JSON: {parsed_nlp}")
                    
                    # 2. Scrape Step
                    scrape_result = solve_with_scraper(
                        gender=parsed_nlp.get("gender", "ذكر"),
                        heirs=parsed_nlp.get("heirs", {}),
                        browser=browser,
                        headless=True
                    )
                    
                    # 3. Format Step
                    reasoning = scrape_result.pop("reasoning", "")
                    answer = scrape_result

                result = {
                    "id": case_id,
                    "source": case.get("source", ""),
                    "question": question,
                    "reasoning": reasoning,
                    "answer_structured": answer,
                }

                checkpoint[case_id] = result
                save_checkpoint(checkpoint_path, checkpoint)
                print("ok")

            except Exception as e:
                print(f"Failed: {e}")
                errors += 1
                checkpoint[case_id] = {
                    "id": case_id,
                    "error": str(e),
                    "raw_snippet": (raw or "") if raw else "",
                }
                save_checkpoint(checkpoint_path, checkpoint)
                time.sleep(provider.retry_delay)
                continue

            time.sleep(1)
            
    finally:
        # Cleanup browser resources cleanly
        if browser:
            browser.close()
        if playwright_context:
            playwright_context.stop()

    # Write final predictions
    order = {case["id"]: idx for idx, case in enumerate(cases)}
    predictions = [v for v in checkpoint.values() if "error" not in v and "answer_structured" in v]
    predictions.sort(key=lambda r: order.get(r["id"], 9999))

    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    success = len(predictions)
    print(f"\nDone: {success}/{total} predictions ({errors} errors)")
    return predictions_path


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QIAS 2026 Al-Mawarith Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--name", default=None, help="Custom name for the output folder")
    parser.add_argument("--run-dir", default=None, help="Reuse an existing output directory")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run failed cases")
    parser.add_argument("--analyze-only", action="store_true", help="Skip prediction, just eval")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_dir = Path(args.config).parent
    method = cfg.get("method", "llm_reasoning")
    bench_cfg = cfg["benchmark"]

    dataset_dir = base_dir / cfg["paths"]["dataset_dir"]
    output_root = base_dir / cfg["paths"]["output_dir"]
    evaluator_src = base_dir / cfg["evaluation"]["evaluator_src"]
    
    # Determine which prompt to load
    if method == "llm_scraping":
        prompt_path = base_dir / cfg["paths"].get("parsing_prompt", "prompts/parsing_prompt.txt")
    else:
        prompt_path = base_dir / cfg["paths"]["system_prompt"]

    # ── Dynamic Folder Resolution ──
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"ERROR: --run-dir {run_dir} does not exist")
            return
    elif args.name:
        run_dir = output_root / args.name
    else:
        model_name = cfg["model"]["name"]
        source = bench_cfg["source"]
        n_samples = bench_cfg["sample_size"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Creates highly organized nested folders: output/method/model/source_n_time/
        run_name = f"{source}_n{n_samples}_{timestamp}"
        run_dir = output_root / method / model_name / run_name
        
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"  QIAS 2026 — Al-Mawarith Pipeline")
    print(f"  Method:   {method}")
    print(f"  Model:    {cfg['model']['provider']}/{cfg['model']['name']}")
    print(f"  Prompt:   {prompt_path.name}")
    print(f"  Source:   {bench_cfg['source']} (n={bench_cfg['sample_size']})")
    print(f"  Output:   {run_dir}")
    print(f"{'='*65}\n")

    # ── Step 1: Load / sample cases ───────────────────────────
    source = bench_cfg["source"]
    all_cases = load_all_cases(dataset_dir, source)
    print(f"Loaded {len(all_cases)} cases from {source}/")

    if source == "train":
        cases = sample_cases(all_cases, bench_cfg["sample_size"], bench_cfg["seed"])
        print(f"Sampled {len(cases)} cases (seed={bench_cfg['seed']})")
    else:
        cases = all_cases
        print(f"Using all {len(cases)} dev cases")

    sample_path = run_dir / "sample.json"
    sample_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    ref_path = run_dir / "reference.json"
    ref_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Step 2: Run predictions ───────────────────────────────
    if not args.analyze_only:
        prompt_text = prompt_path.read_text(encoding="utf-8")
        provider = get_provider(cfg["model"])
        predictions_path = run_predictions(
            provider=provider,
            prompt=prompt_text,
            cases=cases,
            output_dir=run_dir,
            method=method,
            retry_failed=args.retry_failed,
        )
    else:
        predictions_path = run_dir / "predictions.json"
        if not predictions_path.exists():
            print(f"ERROR: No predictions.json found in {run_dir}")
            return

    # ── Step 3: Evaluate ──────────────────────────────────────
    if source == "test":
        print(f"\nTest set — skipping local evaluation (submit predictions for scoring).")
        print(f"Predictions saved to: {run_dir / 'predictions.json'}")
        config_snapshot = run_dir / "config_snapshot.yaml"
        config_snapshot.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        print(f"\nAll outputs saved to: {run_dir}/")
        return

    print(f"\nEvaluating with official MIR-E scorer...")
    summary = evaluate_run(predictions_path, ref_path, run_dir, str(evaluator_src))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEvaluation summary:")
    print(json.dumps(summary, indent=2))

    # ── Step 4: Analyze ───────────────────────────────────────
    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        print(f"\nAnalyzing results...")
        analysis = build_analysis(eval_path)

        analysis_path = run_dir / "analysis.json"
        analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'='*65}")
        print(f"  RESULTS")
        print(f"{'='*65}")
        print(f"  Total cases:    {analysis['total']}")
        print(f"  Perfect (1.0):  {analysis['perfect']}")
        print(f"  Failed:         {analysis['failed']}")
        print(f"  Avg MIR-E:      {analysis['avg_mire']:.4f}")
        print(f"")
        if analysis["worst_10"]:
            print(f"  Worst cases:")
            for c in analysis["worst_10"]:
                print(f"    {c['id']}: {c['mire']:.4f}")
        print(f"{'='*65}")

    config_snapshot = run_dir / "config_snapshot.yaml"
    config_snapshot.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    print(f"\nAll outputs saved to: {run_dir}/")


if __name__ == "__main__":
    main()