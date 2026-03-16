# QIAS 2026 Al-Mawarith Pipeline

System description for the [QIAS 2026 Shared Task](https://sites.google.com/view/qias2026) on Islamic inheritance reasoning.

We compare two approaches: (1) a reasoning-based approach where an LLM solves inheritance problems end-to-end using a detailed Arabic prompt, and (2) a hybrid approach where the LLM extracts heirs and an external rule-based calculator ([Almwareeth](https://almwareeth.com)) computes the shares.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
playwright install
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## External Dependencies

### Dataset

Download the MAWARITH dataset from the [official QIAS 2026 repository](https://gitlab.com/islamgpt1/qias_shared_task_2026) and place it under `dataset/`:

```
dataset/
  train/
    2025/   # qias2025_almawarith_part*.json
    2026/   # qias2026_almawarith_part*.json
  dev/      # qias2025_almawarith_part1.json, part61.json
  test/     # qias2025_almawarith_test_id_question.json
```

### Official Evaluator

The MIR-E evaluation scorer from the [official QIAS 2026 shared task](https://gitlab.com/islamgpt1/qias_shared_task_2026) is bundled under `evaluator/`. No additional cloning is needed. No additional cloning is needed.

## Configuration

All settings are controlled via `config.yaml`. Pre-configured templates are provided in `configs/`:

| File | Model | Method |
|------|-------|--------|
| `gemini_pro_reasoning.yaml` | Gemini 2.5 Pro | Reasoning |
| `gemini_flash_scraping.yaml` | Gemini 2.5 Flash | Scraping |
| `mistral_med_reasoning.yaml` | Mistral Medium 2508 | Reasoning |
| `mistral_small_scraping.yaml` | Mistral Small 2506 | Scraping |

Copy a template to `config.yaml` to use it:

```bash
cp configs/gemini_pro_reasoning.yaml config.yaml
```

Key fields in `config.yaml`:

- `method`: `llm_reasoning` or `llm_scraping`
- `model.provider`: `gemini` or `mistral`
- `model.name`: model identifier (e.g. `gemini-2.5-pro`)
- `benchmark.source`: `train`, `dev`, or `test`
- `benchmark.sample_size`: number of cases to run

## Usage

```bash
# Run the full pipeline (sample, predict, evaluate, analyze)
python run.py

# Run with a custom output folder name
python run.py --name my_experiment

# Re-run only failed cases from a previous run
python run.py --retry-failed --run-dir output/dev/my_experiment

# Skip prediction, just re-evaluate existing results
python run.py --analyze-only --run-dir output/dev/my_experiment

# Delete sub-1.0 MIR-E cases from checkpoint so they get re-run
python scripts/del_failed.py output/dev/my_experiment

# Delete specific case IDs from checkpoint
python scripts/del.py output/dev/my_experiment/checkpoint.json case_id_1 case_id_2
```

## Project Structure

```
├── run.py                    # Main pipeline entry point
├── config.yaml               # Active configuration
├── requirements.txt
├── .env.example
├── configs/                  # Pre-configured templates
│   ├── gemini_pro_reasoning.yaml
│   ├── gemini_flash_scraping.yaml
│   ├── mistral_med_reasoning.yaml
│   └── mistral_small_scraping.yaml
├── prompts/
│   ├── system_prompt.txt     # Arabic reasoning prompt (~23.6k tokens)
│   └── parsing_prompt.txt    # Heir extraction prompt (~3.7k tokens)
├── src/
│   ├── evaluation.py         # Bridge to official MIR-E scorer
│   ├── parsing.py            # LLM response parsing
│   ├── scraper.py            # Playwright scraper for almwareeth.com
│   └── providers/
│       ├── base.py           # Abstract provider + factory
│       ├── gemini.py
│       └── mistral_provider.py
├── evaluator/                # Bundled MIR-E scorer (from official shared task)
└── scripts/
    ├── del.py                # Delete specific cases from checkpoint
    ├── del_failed.py         # Delete failed cases from checkpoint
    ├── convert_predictions.py # Format conversion utility
    └── make_charts.py        # Generate result charts
```

Each run produces an output directory containing: `sample.json`, `reference.json`, `checkpoint.json`, `predictions.json`, `eval.json`, `eval.csv`, `summary.json`, `analysis.json`, and `config_snapshot.yaml`.

## Reproducing Results

All experiments use `temperature: 0`, `seed: 42`, and the full test set of 500 cases. To reproduce each configuration:

```bash
# Gemini 2.5 Pro + Reasoning (MIR-E: 0.92)
cp configs/gemini_pro_reasoning.yaml config.yaml
python run.py --name gemini_pro_reasoning

# Gemini 2.5 Flash + Scraping (MIR-E: 0.99)
cp configs/gemini_flash_scraping.yaml config.yaml
python run.py --name gemini_flash_scraping

# Mistral Medium 2508 + Reasoning (MIR-E: 0.62)
cp configs/mistral_med_reasoning.yaml config.yaml
python run.py --name mistral_med_reasoning

# Mistral Small 2506 + Scraping (MIR-E: 0.94)
cp configs/mistral_small_scraping.yaml config.yaml
python run.py --name mistral_small_scraping
```

Results are saved to `output/` and include per-case evaluation scores in `eval.json` and aggregate metrics in `summary.json`.

The scraping configurations require an active internet connection to reach [Almwareeth](https://almwareeth.com).
