# Getting Started

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key

## Setup

```bash
# Clone the repo and install dependencies
git clone <repo-url> && cd <repo-name>
uv sync
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Optionally choose the model (defaults to `gpt-4o-mini`):

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

## Run a Full Evaluation

This starts both the green agent (evaluator) and the purple agent, runs all
rounds, and prints scores:

```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts uv run python -m agentbeats.run_scenario scenario_openai_purple.toml
```

Add `--show-logs` and `AGENT_DEBUG=1` for verbose output:

```bash
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 \
  uv run python -m agentbeats.run_scenario scenario_openai_purple.toml --show-logs
```

Transcripts are saved to `pragmatic_builder/logs/transcripts/<timestamp>/`.

## Run with Docker

```bash
# Build the purple agent image
docker build -t agentwhetters-purple -f Dockerfile.purple .

# Run it (pass your API key)
docker run -p 9018:9018 -e OPENAI_API_KEY="sk-..." agentwhetters-purple
```

## Run Tests

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```
