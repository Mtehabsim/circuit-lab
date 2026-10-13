# Circuit platform implementation plan

## Product target

Deliver a local, usable MVP where a learner opens a challenge, writes Python experiments against a real small PyTorch checkpoint, inspects measured traces, and submits a structured mechanism claim with causal evidence. The first release focuses on the six agreed foundation challenges. It does not include accounts, social features, payments, or large pretrained models.

## Approved architecture

- **Frontend:** React, TypeScript, and Vite. Monaco provides the Python editor. The challenge brief and examples stay beside the editor; run output, traces, and submission feedback appear below it.
- **API:** FastAPI serves challenge metadata, starts experiment runs, returns results, and accepts submissions.
- **Model layer:** PyTorch checkpoints and deterministic dataset definitions. Challenge checkpoint builders write revisioned files under ignored `var/models/`; builders and challenge metadata are versioned in the repository. A common `ModelAdapter` gives the six small models the same inspection interface; a TransformerLens adapter can support pretrained-model challenges later.
- **Execution:** The API launches learner code in a fresh local Python subprocess with a time limit, bounded output, and a clean temporary working directory. A separate subprocess grades the returned claim. This is a trusted, single-user development setup: local learner code runs with the app user’s operating-system permissions.
- **Database:** SQLite stores anonymous session IDs, runs, submissions, scores, and completion state. Challenge definitions and checkpoint builders stay versioned in the repository. Postgres is a later deployment change if concurrent usage requires it.
- **Challenge contract:** JSON definitions describe the model, task, public data, allowed tools, submission schema, hidden evaluation, scoring, hints, and post-completion explanation.

## Progress since approval

- [x] Add the Vite/React/TypeScript project structure, Monaco editor workspace, challenge brief, run/output/attention/submission panels, and responsive styling.
- [x] Define the previous-token challenge in JSON with separate public and held-out datasets.
- [x] Implement a small PyTorch causal-attention model, checkpoint builder, tokenizer, metrics, attention inspection, and measured head ablation.
- [x] Implement FastAPI challenge/run/submit routes and SQLite persistence for runs and submissions.
- [x] Add session-scoped persisted run retrieval at `GET /api/runs/{run_id}`.
- [x] Add Pydantic-validated, versioned challenge definitions with startup validation and safe challenge-id resolution.
- [x] Add a shared model adapter with named residual, Q/K/V, attention-logit/pattern, head-output, and logit-lens access.
- [x] Version experiment events and worker results; independently replay claimed effects on held-out data.
- [x] Add SQLite-backed drafts, recent run history, challenge completion progress, and post-submission explanations.
- [x] Implement all six MVP challenge models, datasets, intervention tasks, graders, hints, and explanations.
- [x] Run worker and grader locally as bounded subprocesses; terminate the process group on timeout or output overflow.
- [x] Generate frontend and Python lockfiles and verify a clean dependency sync.
- [x] Keep learner and grader entry points in separate subprocesses; pass only public examples into the learner process.
- [x] Add model/intervention/grader test cases and local setup documentation.
- [x] Validate Python source syntax, Pydantic challenge schema loading, SQLite run/submission/draft persistence, worker result parsing, and timeout enforcement.
- [x] Verify local worker output caps and return an actionable missing-PyTorch setup error.
- [x] Install dependencies, compile the frontend, and run API/model tests (20 passed; frontend production build passes).
- [x] Declare NumPy and PyTorch as runtime dependencies and refresh the lockfile so a normal local setup has NumPy available to PyTorch.
- [x] Show per-challenge submission field types and deterministic grading checks, explain which fields are not graded, and document direct PyTorch model calls beside the experiment helpers.
- [x] Run all six reference challenges through FastAPI, the local worker, SQLite, and hidden grading (all score 100/100).

**Environment note:** Dependencies are installed. The selected local subprocess runner does not require Docker.

## Milestone 0 — Make the repository runnable

**Status: first vertical slice is runnable.** Dependencies install, the frontend production build passes, and the full run/submit flow was exercised through FastAPI and returned a 100/100 hidden-test grade.

1. Replace the standalone `index.html` prototype with a Vite React TypeScript app.
2. Add a Python project definition and lockfile for FastAPI, PyTorch, and backend tooling.
3. Add environment configuration and documented commands for starting the frontend, API, and worker.
4. Add a local Python worker and document the trusted single-user execution model.
5. Add a health endpoint and a frontend API connection check.

**Ready when:** a developer can start the app from a clean checkout using documented commands, see API health in the UI, and gets a clear setup error if Python dependencies are unavailable.

## Milestone 1 — Define the challenge and model contracts

**Status: complete for the six-checkpoint MVP.** Pydantic validates versioned challenge JSON at startup. The common `ModelAdapter` provides forward passes and named activation caches for logits, residual states, attention patterns/logits, Q/K/V, and head outputs. `ExperimentTools` adds evaluation, ablation, OV analysis, and patching with versioned experiment events. A TransformerLens adapter for future pretrained checkpoints remains out of scope.

1. Define typed challenge metadata: id, title, difficulty, category, objectives, behavior, public examples, allowed tools, intervention limits, submission schema, hints, and solution notes.
2. Define a versioned JSON challenge format and validate it when the API starts.
3. Define the common Python user entry point:

   ```python
   def solve(model, tokenizer, dataset, tools):
       ...
       return {
           "components": [{"type": "attention_head", "layer": 1, "head": 2}]
       }
   ```

4. Specify a stable `tools` API for evaluation, attention/activation reads, ablation, activation patching, and recording experiments.
5. Implement `ModelAdapter` methods for forward passes and named activation caching; implement ablation and patching as non-mutating forward arguments so cleanup is automatic after each call.
6. Implement versioned run-event records so the grader can distinguish measured interventions from claims written by the learner.

**Ready when:** a challenge definition loads into typed data, a model adapter can run it, and the API returns machine-readable metrics and intervention records.

## Milestone 2 — Build and validate the first real challenge

**Status: first challenge validated.** Checkpoint behavior, held-out accuracy, target-head ablation, and grading replay passed the PyTorch tests and the live API flow.

1. Deterministically construct a small causal transformer for previous-token copying; generate a revisioned checkpoint and tokenizer behavior from the versioned builder.
2. Keep its ground-truth circuit in grader-only files that are not passed to learner processes.
3. Generate public and hidden sequences with unseen tokens, lengths, and positions.
4. Implement previous-token accuracy as the public metric.
5. Verify that the checkpoint has high baseline accuracy and that ablating the target head substantially lowers the metric.
6. Provide a reference solution that runs the same public tools exposed to learners.

**Ready when:** the checkpoint reproduces the behavior from disk, public examples behave as described, and independent validation confirms the expected causal effect across hidden examples.

## Milestone 3 — Implement run and submission services

**Status: local MVP run/submit flow verified.** Challenge loading, local worker launch, session-scoped run retrieval, grading/submission persistence, recent-run history, and worker timeout/output handling pass API tests and end-to-end challenge runs. Runs are synchronous; asynchronous queues and concurrency controls are deferred.

1. Add `GET /api/challenges` and `GET /api/challenges/{id}` for public challenge data.
2. Add `POST /api/runs` to accept challenge id, session id, and source code; start a local worker subprocess and return a run id.
3. Add `GET /api/runs/{id}` for status, stdout/stderr, errors, metrics, and renderable traces.
4. Add `POST /api/submissions` to bind a structured claim to a completed run.
5. Have the grader replay or independently verify recorded interventions on hidden inputs instead of trusting values returned by user code.
6. Persist runs, submissions, scores, and per-dimension feedback in SQLite.
7. Handle timeouts, malformed code, invalid component ids, missing evidence, and worker crashes as visible run states.

**Ready when:** a submission cannot receive causal-evidence credit without a corresponding valid intervention record, and every run can be inspected after completion.

## Milestone 4 — Build the coding workspace

**Status: MVP workspace builds.** The split problem/editor layout loads challenge metadata dynamically, supports challenge navigation, challenge-specific starter code, progressive hints, run/submit, output/attention/OV/patch panels, saved run history, persisted drafts, completion progress, and post-submission explanations. A stop/cancel button and browser-level visual testing remain; runs currently have a 20-second bound.

1. Implement the split layout: problem statement, examples, task objectives, and hints on the left; Monaco editor on the right.
2. Add challenge navigation and difficulty/progression state.
3. Add run, stop, reset, and submit controls with clear disabled/loading/error states.
4. Show stdout, stderr, metric deltas, attention heatmaps, activation summaries, and intervention timelines in result tabs.
5. Add a starter `solve(...)` function and short inline API reference for each challenge.
6. Keep edits local until submission; persist anonymous progress and drafts in SQLite through a local session id.
7. Show progressive hints and explanations only at the intended point in the challenge flow.

**Ready when:** the learner can edit Python, run it against the selected checkpoint, see the actual returned results, submit a structured claim, and receive a score with evidence-based feedback.

## Milestone 5 — Add the remaining five MVP challenges

**Status: all six MVP challenge types implemented and end-to-end reference-graded.** The induction and IOI checkpoints use deterministic small circuits, and all graders replay causal checks on hidden sequences/name pairs.

Implement and validate each challenge as its own checkpoint, data generator, grader, and explanation:

1. **Necessary or noise?** Identify a behaviorally necessary head among correlating candidates using ablation.
2. **What gets written?** Trace a known attention head’s value/output contribution to target logits.
3. **Complete the pattern.** Find and causally test a two-head induction circuit.
4. **Find the signal.** Locate a decisive activation with clean/corrupted activation patching.
5. **Who gets the book?** Explain a small IOI behavior with a compact circuit claim and interventions.

For each challenge, define public examples, held-out variations, allowed interventions, score dimensions, progressive hints, a canonical explanation, and regression checks for the checkpoint itself.

**Ready when:** all six challenges have independent ground truth, repeatable hidden evaluation, and feedback that explains missing or weak causal evidence.

## Milestone 6 — Hardening and MVP release

**Status: local trusted-use MVP ready; public deployment hardening remains.** Challenge schema validation, time/output/source limits, session ownership checks, persistence tests, lockfiles, README, and full challenge run/submit verification are complete. Local Python execution is not a sandbox, so this build is suitable only for trusted single-user use.

1. Document the trusted local execution boundary and keep public challenge input separate from grader input.
2. Limit run duration, output size, and input size; terminate worker process groups and always remove temporary files. Concurrent job limits remain future work for this synchronous local runner.
3. Add API and UI coverage for successful runs, syntax errors, timeouts, failed interventions, invalid submissions, and successful submissions.
4. Add reproducible model-generation and challenge-validation commands with pinned seeds and dependency versions.
5. Add a README for local setup, architecture, adding a challenge, and known limitations.
6. Review copy so visual evidence is described as evidence and never presented as a causal explanation by itself.

**Ready when:** a clean local setup can complete a challenge from opening the problem through hidden evaluation, and failure cases return actionable messages without leaving worker processes or temporary artifacts behind. The local trusted-use criteria have been exercised across all six reference solutions; public deployment is not ready without an OS sandbox and concurrency controls.

## Initial build order

Build one vertical slice through Milestones 0–4 using only the previous-token challenge. Once that full run/inspect/intervene/submit path works, add challenges 2–6. This keeps the first review focused on whether the coding loop works before multiplying challenge content.

## Known setup dependency

Learner Python runs as the local app user and is not sandboxed. Use this mode only for trusted local development; a public multi-user deployment needs an operating-system sandbox or container runtime.
