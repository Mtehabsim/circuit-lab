# Circuit

LeetCode-style mechanistic interpretability challenges. The MVP uses a React/TypeScript/Vite client, a FastAPI service, SQLite attempt storage, small PyTorch checkpoints, and local Python worker processes.

## Local setup

Requirements: Node.js 20+, Python 3.11+, and `uv`.

Start the API in a terminal:

```sh
cd backend
uv sync --extra dev
uv run uvicorn circuit_api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```sh
cd frontend
npm install
npm run dev
```

Open the Vite address shown in the frontend terminal. The page uses the Vite `/api` proxy to connect to FastAPI. Run the starter code, use the experiment helpers to record interventions, return the required structured component claim, and select **Submit** to grade it on held-out sequences.

## Challenge entry point

Each run invokes `solve(model, tokenizer, dataset, tools)` in a fresh local Python subprocess. The process has a time limit and bounded captured output; the grader runs in a separate subprocess and replays the claimed intervention against held-out examples.

The first challenge's target circuit is a deterministic PyTorch checkpoint. The grader process evaluates held-out data. Local execution is intended for a trusted, single-user development environment: learner Python runs with the same operating-system permissions as the app, so it can inspect local files and hidden tests are not secret from the machine owner. `backend/tests/` covers behavior, attention, interventions, and grading.

## Challenges and experiment tools

The six challenge definitions are versioned JSON files under `backend/circuit_api/challenges/`. The current curriculum includes previous-token localization, causal ablation with a correlated decoy, an OV/value-path trace, a two-head induction circuit, clean-to-corrupted activation patching, and an IOI-style name-mover circuit. Each definition includes public and held-out examples, starter code, hints, submission schema, and an explanation. The API validates challenge versions and progression order at startup.

The shared `ModelAdapter` and `tools` object expose behavior evaluation, named residual/Q/K/V/attention/logit activations, logit-lens reads, head ablation and sweeps, value/output contribution tracing, and head-output activation patching. Submissions are graded by challenge-specific code that replays intervention checks on held-out examples. SQLite keeps drafts, recent runs, and submissions associated with a browser-local anonymous session.

This is a local research/learning MVP, not a public code-execution service. Learner Python runs as the current OS user and can read or change local files. The worker has timeout and output caps, but it is not a security sandbox. Do not expose it to untrusted users without adding an OS-level sandbox.

Authentication, profiles, payments, and social features are out of scope.
