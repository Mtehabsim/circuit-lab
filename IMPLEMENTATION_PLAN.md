# CircuitLab implementation status

## Palindrome Machine: current state

- [x] Independently generated task data and seeded, trained checkpoint with recorded provenance.
- [x] Monaco/Python research editor, runnable scripts, bounded outputs, Matplotlib artifacts, and separate Run/Submit flows.
- [x] Activation cache for attention patterns/logits, Q/K/V, head outputs, residuals, and MLP activations. `run_with_cache()` now reuses one model trace.
- [x] Independent reference investigation covering controlled behavior probes, QK pair statistics, attention, head/MLP ablations, matched component/control patching, and readout-state localization. Reproduce with `cd backend && uv run python ../docs/research/palindrome_experiments.py`.
- [x] Six sequential milestones with 10/15/15/25/20/15 point weights: behavior metric, attention statistic, causal ablation, raw QK/OV tensor operations, target/control component patching, and hidden comparison against an alternative.
- [x] Private replay checks learner-defined functions on held-out model tensors and synthetic tensors. Learner-reported effect numbers are not used as measurements.
- [x] Explicit reference reveal; unsuccessful submission does not reveal the walkthrough. Early reveal is recorded distinctly from independent completion.
- [x] Journal stores code/output/events/figures and optional hypothesis, expected result, and notes; two runs can be compared.
- [x] Adversarial tests for shallow head sweeps, wrong IDs, broad intervention lists, fabricated metrics, final-residual replacement, tensor formula correctness, false predictions, and an alternative MLP hypothesis.
- [x] Backend suite and frontend production build validated (see the implementation commit and test output).

## Scientific limits

The measured checkpoint supports a partial account: L0H0/L0H1's QK scores at selected mirrored pairs respond to a controlled mismatch; those heads and MLP0 causally affect task accuracy; component-output patches at the changed token move the class margin more than a paired position control but rarely restore the label; late EOS residual patching restores the decision more often and is only readout localization. This does not establish a complete QK/OV/pathway circuit. More granular token-versus-position QK decomposition and path patching remain open research work. Scores verify specific code and intervention contracts, not unique prose explanations.

The current editor starts a fresh bounded Python subprocess for each run. The journal preserves experiments, but variables do not persist across runs. A persistent kernel is secondary to the scientific milestone work and requires kernel lifecycle and stronger isolation.

Local execution and grading are trusted-user development processes, not OS/container security sandboxes. Do not expose them to untrusted public users. Hosted execution requires a separate isolated runner with filesystem/network controls, dependency policy, resource limits, and private grading assets unavailable to learner code.

## Next work

Do not add another algorithmic challenge until the Palindrome mission has been reviewed in its running form. Remaining depth work is to independently evaluate whether a more informative training seed/configuration yields a tractable but genuinely learned mechanism, and to extend the existing milestone contracts only where a claim can be quantitatively replayed. A future First Unique Character mission should use a separately trained model and the same provenance/research-quality gates.

## Curriculum alignment

See [`docs/ARENA_SKILL_COVERAGE.md`](docs/ARENA_SKILL_COVERAGE.md) for the seven-problem skill map, original-content boundary, and current Palindrome milestone verification.
