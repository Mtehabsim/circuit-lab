# CircuitLab

CircuitLab is a code-first mechanistic interpretability research lab. A mission gives the learner a trained model and asks them to write Python experiments, inspect activations, test interventions, and submit a measurable causal claim.

The current active mission, **The Palindrome Machine**, investigates an independently trained transformer classifier. The older hand-constructed toy challenges remain in the repository as archived examples and are not shown in the mission list.

## Run locally

Requirements: Node.js 20+, Python 3.11+, and `uv`.

Start the API:

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

Open the Vite address shown in the frontend terminal. In the editor, **Run** executes an ordinary Python research script in a fresh process. PyTorch, NumPy, Matplotlib, the model, tokenizer, public dataset, and replayable intervention helpers are available. Figures made by the script appear in run output. Set a `submission` dictionary at top level, or define `solve(model, tokenizer, dataset, tools)`, when you want to submit. Submission is separate from running experiments.

Each run is saved with its code, output, events, figures, and optional hypothesis/expected-result/notes in the research journal. You can restore a run or compare two runs side by side. Python variables do not persist between runs yet; keep reusable functions in the script and rerun it. A persistent kernel remains a follow-up because it needs stronger session isolation and lifecycle controls.

## Palindrome checkpoint

The checkpoint was trained from scratch on independently generated palindrome and non-palindrome sequences. Its training code, fixed seed/configuration, weights, and validation provenance are in `backend/circuit_api/palindrome.py` and `backend/circuit_api/checkpoints/`. It achieves 98.9% on the seeded generated validation sample, 100% on the fixed public sample, and 97.9% on the fixed hidden sample (lengths 3–12). An OOD probe at lengths 13–16 is about 50%, so those lengths are outside its supported range.

Regenerate the checkpoint with:

```sh
cd backend
uv run python -m circuit_api.train_palindrome --output circuit_api/checkpoints/palindrome-machine.pt
```

The API copies the bundled trained checkpoint into its local model cache on first use. The model exposes logits and named activation caches including attention patterns/logits, Q/K/V, head outputs, residual streams, and MLP activations. Palindrome now has six sequential milestones: behavior measurement (10), attention statistic (15), bounded causal ablation (15), learner-written QK/OV reconstruction (25), component patching against a matched position control (20), and a hidden comparison against an alternative mechanism (15). A head sweep and successful ablation can earn at most 40/100 across the first three stages. The grader replays metrics; a final-residual patch only localizes the decision. See the [reference investigation](docs/research/PALINDROME_REFERENCE_INVESTIGATION.md) and reproduce its measurements with `cd backend && uv run python ../docs/research/palindrome_experiments.py`.

## Curriculum and ARENA reference

`docs/ARENA_SKILL_COVERAGE.md` maps ARENA's seven public Monthly Algorithmic Problems to future CircuitLab skills and verification. CircuitLab borrows the task category and educational inspiration, but distributes original challenge wording, generated data, model training code, checkpoints, tests, and research walkthroughs. It does not bundle ARENA notebooks, solutions, datasets, or pretrained checkpoints. Review ARENA's [materials-use policy](https://www.arena.education/using-our-materials) before organizing a program based on its materials; organizations are asked to request permission.

`docs/MIGRATION_PLAN.md` records the repository audit, completed work, remaining milestones, and release limits.

## Execution limits

The local worker and grader run in separate subprocesses with time and output bounds. This is a trusted, single-user development setup, not a security sandbox. Learner Python runs with the app user's permissions and can inspect local files, including hidden test files. Do not expose the current worker to untrusted users without an OS/container sandbox, filesystem and network restrictions, dependency controls, and concurrency limits.

Run backend tests with `cd backend && uv run pytest`; build the frontend with `cd frontend && npm run build`.
