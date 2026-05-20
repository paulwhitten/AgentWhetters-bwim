# AgentWhetters BWIM

> **1st place (tied) — Gaming Agents track, AgentBeats competition**
> Leaderboard: [agentbeats.dev/agentbeater/build-what-i-mean](https://agentbeats.dev/agentbeater/build-what-i-mean)

An LLM agent for the **Build What I Mean (BWIM)** benchmark: natural-language-guided 3D construction under ambiguous, underspecified instructions. The agent reads a builder's intent, decides whether it has enough information to act, and either asks a clarifying question or executes the build.

Built on the [A2A (Agent-to-Agent) protocol](https://a2a-protocol.org/latest/) for the [AgentBeats](https://agentbeats.dev) platform. Uses `gpt-4o-mini` with a modular skills pipeline that offloads spatial computation to deterministic code. Adapted from the [upstream BWIM baseline](https://github.com/ltl-uva/build_what_i_mean).

![Figure 1 — A benchmark round requiring T-shape recognition and extension.](https://arxiv.org/html/2605.07066v1/x1.png)

*Figure 1 — A benchmark round requiring T-shape recognition and extension.*

The above figure depicts an initial build state provided to the agent on the left in the form of block coordinates and colors. The agent is then presented with the instruction: 'Keeping the T shape, extend the existing green structure by adding two green blocks to the longer base. Then add one purple block to each arm.' The diagram on the right is a visualization of the resulting output. New blocks are marked with +.

## Approach

The agent combines six techniques to handle BWIM's ambiguity-under-time-pressure setting:

- **Spatial reasoning** — a deterministic spatial executor computes all vertical placement from horizontal coordinates and column occupancy, eliminating y-coordinate errors entirely.
- **Instruction decomposition** — the LLM decomposes each instruction into a plan of typed JSON actions, each specifying action type, horizontal position, color, and count.
- **Underspecification detection** — expected-value analysis flags missing color or count and decides when to ask a clarifying question, with a decision threshold at p\* = 0.75.
- **Adaptive prompt enrichment** — detects difficult spatial concepts (e.g., T-shape, L-shape) and injects targeted enrichments such as few-shot examples to guide the LLM.
- **Peephole optimization** — scans planned actions against pattern-matching rules to catch and correct common LLM errors specific to spatial build tasks before execution.
- **Build space simplification** — a 2.5-D decomposition reduces the output space from |𝒢|×|𝒞| to |{0,…,8}|² × |𝒞|, letting the planner reason in 2D while the executor handles vertical placement.

![Figure 2 — 2.5-D decomposition: the LLM planner generates 2D plans with (x,z) coordinates and action types. The deterministic executor computes vertical placement via column occupancy.](https://arxiv.org/html/2605.07066v1/x2.png)

*Figure 2 — 2.5-D decomposition architecture.*

Detailed pipeline: [purple agent README](./pragmatic_builder/purple_openai/README.md). Article draft: [agentwhetter_bwim_article.pdf](https://paul.whitten.dev/agentwhetter_bwim_article.pdf).

## Results

- **Leaderboard**: [agentbeats.dev/agentbeater/build-what-i-mean](https://agentbeats.dev/agentbeater/build-what-i-mean) — 1st place (tied)
- **Paper** (submitted to NAECON 2026): [arxiv.org/abs/2605.07066](https://arxiv.org/abs/2605.07066)
- **Structural accuracy**: 94.6% with GPT-4o-mini on BWIM (vs. 76.3% best competing system)
- **Ablation**: removing 2.5-D decomposition drops accuracy by 50.7 points to 43.8%
- **IGLU transfer**: block-level F1 improves from 0.723 to 0.798 across 500 tasks
- **Provisional patent** filed April 2026

## Project Structure

```
pragmatic_builder/
├─ builder_agent.py   # Main server entrypoint + agent card
├─ green_agent.py     # Agent logic
├─ evaluator_proxy.py # Proxy server for evaluation flows
├─ agentbeats/        # AgentBeats integration helpers
└─ skills/            # Skills used to solve BWIM instructions
data/                 # Scenario data files
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
.github/workflows/test-and-publish.yml # CI workflow
```

## Running Locally

```bash
# Install dependencies
uv sync

# Run the builder agent (purple agent dummy)
uv run pragmatic_builder/builder_agent.py --host 127.0.0.1 --port 9019

# Run the green agent (evaluation)
uv run pragmatic_builder/evaluator_proxy.py --host 127.0.0.1 --port 9009
```

### Default scenario

```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 \
    uv run python -m agentbeats.run_scenario scenario.toml --show-logs
```

### Scenario with a questionnaire

```bash
cd pragmatic_builder
AGENT_QA_MODE=dummy AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 \
    uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Scenario with OpenAI QA

```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
AGENT_QA_MODE=openai AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 \
    uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Scenario with an OpenAI purple agent

```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 \
    uv run python -m agentbeats.run_scenario scenario_openai_purple.toml --show-logs
```

### Scenario agents + CLI client (writes results.json)

```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts \
    uv run python -m agentbeats.run_scenario scenario.toml --serve-only &
```

```bash
cd pragmatic_builder
uv run python -m agentbeats.client_cli scenario.toml results.json
```

## Running with Docker

```bash
# Build the green agent image
docker build -t my-agent-green -f Dockerfile .

# Build the purple agent image
docker build -t my-agent-purple -f Dockerfile.purple .

# Run the green agent (evaluation)
docker run -p 9009:9009 my-agent-green

# Run the purple builder agent
docker run -p 9018:9018 my-agent-purple
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image to GitHub Container Registry.

- **Push to `main`** → publishes `latest`: `ghcr.io/<user>/<repo>:latest`
- **Git tag** (e.g. `v1.0.0`) → publishes version tags: `ghcr.io/<user>/<repo>:1.0.0`, `:1`

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets.

> Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/).
