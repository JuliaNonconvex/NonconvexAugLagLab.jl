# NEWS

## 0.1.4

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.3 and made the package uninstallable on
  Julia 1.10/1.11 (users silently resolved to 0.1.2).
- Requires Julia 1.10+.

## v0.2.3

- **TRON inner solver (`sub_options=(; linesearch=:tron)`, opt-in):** a
  projected-CG trust-region step with active-set identification on the
  bound-constrained AL subproblem. ~32% faster than the default
  `:trustregion` on the TopOpt local-stress benchmark; `:trustregion`
  remains the default.
- **`warm_continue` option:** reuse the previous run's final iterate,
  multipliers (rescaled to the new constraint scales), per-group penalty
  weights and quasi-Newton curvature when `optimize!` is called again on the
  same workspace - e.g. across SIMP continuation stages. `ρ`, `η`, `ω`, the
  filter and the Anderson history re-initialize from the new options; a
  structure mismatch falls back to a cold start.
- **Warm-started L-BFGS across AL outer iterations** (`memory=50`,
  `reset=false`), penalty-scaled diagonal preconditioner, Fletcher-Leyffer
  filter globalization, Anderson acceleration (depth 2), Levenberg damping,
  stall retry, selective penalty weights, safeguarded multipliers, elastic
  mode (opt-in).
- NonconvexIpopt compat widened to `0.7, 0.8`.

## v0.2.2

- **Scaling (default on):** the relaxed constraints are normalized by their
  gradient magnitudes and the objective by its initial magnitude
  (Curtis–Jiang–Robinson style), so the penalty applies to balanced,
  unit-scaled violations. This removes the ill-conditioning that made the
  classical scheme diverge on problems like AC-OPF.
- **Least-squares multiplier initialization is now the default** for equality
  constraints.
- **Blow-up guard:** subproblem iterates with non-finite or exploding
  violations are discarded and the penalty is escalated instead of poisoning
  the multipliers.
- **Thorough built-in subsolver defaults:** the quasi-Newton subproblem solver
  now defaults to `max_iter = 800, tol = 1e-8` (configurable through
  `sub_options`).
- **Two-way penalty update** (Curtis et al.) backs the penalty down once
  feasibility is nearly attained.
