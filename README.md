# Building Game Task - Quick Guide
Based on the minimal template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) green agents compatible with the [AgentBeats](https://agentbeats.dev) platform.

## Project Structure

```
pragmatic_builder/
├─ builder_agent.py   # Main server entrypoint + agent card
├─ green_agent.py     # Agent logic
├─ evaluator_proxy.py # Proxy server for evaluation flows
└─ agentbeats/        # AgentBeats integration helpers
data/                 # Scenario data files
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```
## How to Play

### Running Locally

```bash
# Install dependencies
uv sync

# Run the builder agent (purple agent dummy)
uv run pragmatic_builder/builder_agent.py --host 127.0.0.1 --port 9019

# Run the green agent (evaluation)
uv run pragmatic_builder/evaluator_proxy.py --host 127.0.0.1 --port 9009
```

### Running the default Scenario
```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario.toml --show-logs
```

### Running a Scenario with a questionnaire
```bash
cd pragmatic_builder
AGENT_QA_MODE=dummy AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Running a Scenario with OpenAI QA
```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
AGENT_QA_MODE=openai AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Running a Scenario with an OpenAI Purple Agent
```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_openai_purple.toml --show-logs
```

### Run Scenario Agents + CLI Client (writes results.json)
```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts \
  uv run python -m agentbeats.run_scenario scenario.toml --serve-only &
```

```bash
cd pragmatic_builder
uv run python -m agentbeats.client_cli scenario.toml results.json
```

## Running with Docker (not tested yet)

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

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
