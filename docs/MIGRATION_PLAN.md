# CircuitLab implementation plan

## Product target

Build a standalone, code-first research challenge platform inspired by the skills exercised in ARENA. Reuse only the existing CircuitLab infrastructure where it supports genuine model investigation. All distributed mission content, training data, checkpoints, tests, and reference analyses are independently created. See `ARENA_SKILL_COVERAGE.md` for source boundaries and skill mapping.

## Existing architecture audit

Keep the React/TypeScript/Monaco coding workspace, FastAPI routes, SQLite drafts/history, JSON challenge definitions, PyTorch adapter, worker/grader subprocesses, output/time limits, and hidden data separation. Improve the worker so learner code can run iterative experiments and render its own plots. Refactor exact component-ID graders. Replace hand-constructed circuits as the core algorithmic curriculum with trained checkpoints and explicit provenance. Local execution still runs as the developer's OS user and is not safe for public untrusted submissions.

## Priority plan

### 1. ARENA study, repository audit, and scope — complete
- [x] Inspect challenge schemas, adapters, execution protocol, frontend, persistence, grader, and tests.
- [x] Review ARENA's monthly-problem introduction and palindrome task statement, including model/task structure and the stated standard for a full research solution.
- [x] Review ARENA's materials policy and document an original-content boundary.
- [x] Map seven algorithmic challenges to learner code and objective verification.

### 2. Palindrome mission — implemented; browser review remains
- [x] Add a reproducible independently authored palindrome data generator and training/checkpoint code.
- [x] Train and bundle a checkpoint with recorded seed/configuration; validation is 98.9%, fixed public accuracy 100%, and fixed hidden accuracy 94/96 (97.9%). Lengths 13–16 are about chance and outside the trained range.
- [x] Add model adapter/cache support for attention, Q/K/V, head outputs, residuals, MLP activations, and weights.
- [x] Replace the active previous-token challenge with the staged Palindrome Machine research mission.
- [x] Add learner-authored Python plots and bounded figure output; preserve source, stdout, events, and artifacts in run history.
- [x] Add independently replayed ablation and patch grading, hidden-set scoring, intervention-size limits, and explicit limitations on prose-mechanism grading.
- [x] Write an original checkpoint-specific reference investigation with reproducible code and measured limitations.
- [x] Test API open/run/plot/submit/feedback flow; build the frontend production bundle.

### 3. First Unique Character — queued for next iteration
- [ ] Define a separate original data generator and trained model.
- [ ] Validate its actual learned behavior and OOD limits.
- [ ] Design novel prefix-state investigations and causal tests rather than reusing palindrome milestones.

### 4. Later curriculum
- [ ] Add the other five algorithmic model missions after the first two pass their quality gates.
- [ ] Add foundation missions for attention implementation, hooks, probes, SAE features, IOI, patching, and path analysis where sequencing needs them.
- [ ] Add appropriately licensed pretrained checkpoints only after auditing each asset's source and license.

### 5. Public deployment readiness
- [ ] Isolate Python with an OS/container sandbox, dependency allowlist, filesystem/network controls, and concurrency limits.
- [ ] Keep hidden tests out of learner-visible process and test that the grader ignores learner-printed metrics.

## Quality gates for each mission

A mission is complete only when (1) training is reproducible and checkpoint provenance is recorded, (2) stated behavior is measured and its limits documented, (3) the learner can write/run real Python experiments and plots, (4) interventions are replayed independently on private data, (5) destructive interventions cannot earn full credit, (6) reference analysis describes the actual checkpoint, and (7) a full learner flow and meaningful tests pass. The Palindrome mission passes automated API-flow tests and the frontend build; direct browser review remains useful because no interactive browser was available in this session.
