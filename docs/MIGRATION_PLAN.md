# CircuitLab migration and release status

## Existing architecture retained

The React/TypeScript/Monaco editor, FastAPI API, SQLite drafts/history, JSON challenge configuration, PyTorch model adapter, worker/grader subprocesses, output/time bounds, and original trained Palindrome checkpoint remain. The older hand-designed circuit challenges are archived examples; the active mission is the trained Palindrome Machine. The exact-ID grader has been replaced for this mission with staged Python-function checks and independent intervention replay.

## Palindrome mission: implemented

- Independently authored data generation, training code, checkpoint, and provenance.
- Real model access to weights and activation tensors; `run_with_cache()` reuses the forward trace.
- Python Run/Submit separation, generated figures, tracebacks, and local research journal with run comparison.
- Six gated milestones with 100 total points; head localization and ablation alone cap at 40.
- Grader executes learner-defined functions on hidden model tensors and synthetic tensors, replays ablations and matched component/control patches, and checks candidate-versus-alternative predictions. It ignores learner-printed metrics.
- Final EOS residual replacement is explicitly treated as decision localization; it cannot pass the upstream patch milestone.
- Reference walkthrough and measurement script report both findings and unresolved mechanism questions.
- Reference explanation is removed from the learner-run worker payload and is returned only after all milestones pass or explicit reveal. Reveal marks the mission as revealed rather than independently completed.
- API route, shortcut/false-evidence tests, full milestone journey simulation, and production frontend build pass.

## Current limitations and follow-up

- Python globals restart on each Run. Code, stdout, artifacts, interventions, and optional research notes persist, but live kernel state does not.
- This checkpoint has not been fully reverse-engineered. Early mirrored QK sensitivity, MLP contribution, interaction, and patch evidence are measured; semantic QK decomposition and a complete causal path remain unestablished.
- The learner function is evaluated inside a bounded local grader subprocess with hidden test data. The subprocess is not an OS security boundary; a determined local user can inspect accessible files or process state. Do not expose this setup to public or untrusted users. A hosted version needs OS/container isolation, a restricted filesystem/network, resource and concurrency limits, and grading assets separated from learner execution.
- Do not add another algorithmic mission until the Palindrome experience has been reviewed. Subsequent models need separate training provenance and investigation of the actual learned checkpoint.

## Source and content boundary

ARENA's public curriculum is a reference for research skills. CircuitLab's wording, generated dataset, model, tests, grader, and reference analysis are original. Read [`ARENA_SKILL_COVERAGE.md`](ARENA_SKILL_COVERAGE.md) and the linked [materials-use policy](https://www.arena.education/using-our-materials) before any future use of ARENA assets.
