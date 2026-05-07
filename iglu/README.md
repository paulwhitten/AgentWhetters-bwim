# IGLU Transfer Experiment

Evaluates the 2.5-D decomposition on the IGLU collaborative building dataset
to test generality beyond the primary BWIM benchmark.

## Requirements

- Python 3.11+
- OpenAI API key in `.env` (or exported as `OPENAI_API_KEY`)
- IGLU single-turn dataset CSV at `iglu-datasets/datasets/iglu_single_turn.csv`

Install dependencies from the project root:

```bash
uv sync
```

## Usage

```bash
# Bare LLM baseline (no decomposition)
uv run python iglu/evaluate.py --mode bare --gravity-only --limit 500

# 2.5-D decomposed prompt
uv run python iglu/evaluate.py --mode decomp25d --gravity-only --limit 500
```

### Options

- `--mode bare|decomp25d`: Bare coordinate-output prompt or 2.5-D decomposed prompt.
- `--gravity-only`: Filter to tasks where all blocks satisfy the gravity constraint.
- `--additive-only`: Further filter to additive-only tasks (no removals).
- `--limit N`: Evaluate the first N tasks after filtering.
- `--model MODEL`: OpenAI model name (default: `gpt-4o-mini`).

Results are written to `iglu/results_{mode}_{model}_{limit}_{filter}.json`.

## Evaluation metric

Block-level F1 with maximal intersection over 4 rotations and all valid
translations within the build zone. This accounts for the fact that IGLU
instructions often do not specify absolute position.

## Paper reference

Results from this script appear in Section V-F of the NAECON 2026 paper.
