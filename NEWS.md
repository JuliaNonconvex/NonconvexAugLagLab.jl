# NEWS

## Unreleased

- **First-subproblem tightening (new default)**: `AugLagOptions(first_sub_tol
  = 1e-6)` solves the first subproblem to `min(sub_tol, first_sub_tol)` —
  the first inner solve selects the local basin the whole solve follows.
  Measured on the TopOpt local-stress single-level p=3 solve: the default
  `:eta` schedule plus this tightening reproduces the light-design
  trajectory of a fixed-`1e-6` subproblem tolerance bit-for-bit at 12.7%
  less wall-clock (264.6 vs 303.1 s, 11 outer iterations, obj 0.0938 vs
  0.0976 for the untightened schedule). Set `first_sub_tol = nothing` for
  the previous behavior.
- **Projected-gradient stopping for the built-in quasi-Newton solver (new default)**:
  `QuasiNewtonAlg(projected_stop = true)` terminates on the box-projected
  gradient norm — the correct stationarity measure when the optimum sits on
  a box bound, where the raw gradient does not vanish and the solver
  previously burned its iteration budget — and reports that projected norm
  as the KKT residual. Measured on the TopOpt local-stress single-level
  p=3 solve (80×20, 1600 stress constraints): bit-for-bit identical
  trajectory and solution at 7.2% less wall-clock (303.1 vs 326.7 s,
  11 outer iterations). Set `projected_stop = false` for the previous
  behavior.
- **Anderson acceleration on the inequality multipliers (opt-in)**:
  `AugLagOptions(anderson_ineq = true)` extends the depth-2 Anderson
  mixing from the equality block to the inequality multiplier map of the
  classical quadratic penalty (the unprojected update is mixed, the mix
  re-projected onto λ ≥ 0). Measured on the TopOpt local-stress
  single-level solve: 4.4% slower (316.3 vs 303.1 s) but lands on a 4.6%
  lighter feasible design (obj 0.0895 vs 0.0938) at the same max solid
  stress and a marginally larger final violation (6.2e-6 vs 5.7e-6) — a
  quality-for-time trade, off by default.
- **Practical relative subproblem tolerance (opt-in)**:
  `AugLagOptions(sub_tol_schedule = :relative, sub_tol_rel = τ)` sets the
  inner tolerance proportional to the current constraint violation
  (`clamp(τ·v_prev, 1e-10, 1e-2)`, Eckstein–Silva). Measured on the TopOpt
  local-stress single-level solve: bit-for-bit the same trajectory as the
  `:eta` schedule at 3% more wall-clock (290.9 vs 282.2 s) — no win, so
  `:eta` stays the default. The duplicate `_set_sub_tol`/`_sub_tol_value`
  definitions inside `optimize!` were removed along the way (the inner
  copies shadowed the module-level ones).
- **Relaxed multiplier step (opt-in)**: `AugLagOptions(multiplier_relax =
  true)` scales each dual increment by the achieved-accuracy ratio
  `θ = clamp(sub_tol/‖res‖, 0.1, 1)` of the subproblem stationarity
  residual (Eckstein-style adaptive relaxation), so loose inner solves
  cannot poison the multipliers with a full step. Measured on the TopOpt
  local-stress single-level solve: 2.1% slower (309.5 vs 303.1 s) but a
  2.1% lighter feasible design (obj 0.0918 vs 0.0938) with marginally
  better feasibility — a small quality-for-time trade, off by default.

## 0.4.0

- **Adaptive subproblem tolerance for every sub-solver** (default): the new `sub_tol_schedule = :eta` loosens the inner tolerance as the outer feasibility target tightens (`clamp(η², 1e-6, 1e-2)`), 28% faster than the old schedule on TopOpt stress with 4.8× tighter final feasibility. External sub-solvers receive the per-iteration tolerance through their options via the new package extensions (`_set_sub_tol`): Ipopt (`tol`), MMA (`Tolerance`) and NLopt (`ftol_abs`/`xtol_abs`).
- **`sub_options.tol` is now honored for the built-in QN** (previously silently ignored) and overrides the schedule.
- **Exact Hessians for plain ALM subproblems on JuMP models**: with the Ipopt sub-solver, the subproblem Lagrangian Hessian is assembled from the MathOptInterface evaluator (requires NonconvexIpopt 0.9 and NonconvexCore 2.2.8).
## 0.3.0

- **Proximal regularization (opt-in)**: `AugLagOptions(prox = τ)` adds
  `τ/2‖x − x̄‖²` to the AL subproblems. 10% faster on Rosetta case5 with
  slightly better feasibility (151.7 vs 169.2 s, viol 0.584 vs 0.599); on
  TopOpt local-stress the single-level solve is 32% faster and more
  feasible but converges to a heavier design (anchor bias), while the
  3-stage SIMP continuation wins outright: 2.7× faster (204 vs 560 s) and
  a lighter final design (obj 0.186 vs 0.187) at viol 5.3e-9 — opt-in.
  Prox composes with every penalty mode and with `curvature = :sr1`; the
  `curvature = :newton` subproblems now add the τI term to their exact Hessians.
  Combinations measured on Rosetta case5: prox+PBM is 13% faster with a
  lower f but less feasible (159.5 s, viol 0.524 vs 183.4 s, 0.399);
  prox+SR1 is 27% faster but far from the optimum (f = 14187, viol 0.158
  vs f = 17576, viol 0.016) — not recommended.
- **L-SR1 subproblem curvature (opt-in)**: `AugLagOptions(curvature = :sr1)`
  uses limited-memory symmetric rank-one for the trust-region subproblems.
  Massive Rosetta case5 win (80 vs 219 s, viol 0.016 vs 0.60) but slower on
  TopOpt — opt-in, L-BFGS stays the default.
- **Auto rho_init calibration (opt-in)**: `AugLagOptions(auto_rho_init = true)`
  calibrates the initial penalty from ‖∇f(x0)‖ / ‖c̃⁺(x0)‖. Wins Rosetta
  case5 (viol 0.34 vs 0.60 at 200 outer iterations) but loses TopOpt
  (near-feasible starts → over-large penalty); opt-in, off by default.
- **Ipopt-style restoration trigger (opt-in)**: `AugLagOptions(restore = true,
  restore_stall = k, restore_tol = t, restore_tol_mode = :relative|:absolute,
  restore_prox = τ)` enters the feasibility-restoration phase not only on
  subproblem blow-ups (existing `restore`) but also when feasibility
  stalls: the violation improves by less than `restore_tol` (relative or
  absolute) over `restore_stall` outer iterations. The restoration
  minimizes the raw squared violations over the box with the built-in
  quasi-Newton driver, in the formulation-scaled variables (diagonal
  inverse premultiplier from the JᵀJ diagonal), with an optional
  per-variable pseudo-Huber anchor (`restore_prox`); the restored point
  replaces the iterate, the equality multipliers are re-estimated there,
  and the most-feasible point (restoration or final iterate) is reported.
  Measured on Rosetta AC-OPF case5: viol 0.60 → 1.6e-6 with f = 17858
  (the true optimum) at ~1.8× the default time — a decisive opt-in win
  (both tolerance modes). On the TopOpt SAND tutorial it fires but cannot
  close the equilibrium gap (first-order KᵀK conditioning wall; the
  nested solve is the documented NAND fix); `restore_stall = 0` (default)
  keeps the blow-up-only trigger.
- **Breaking (unreleased branch)**: the `QuasiNewtonAlg` step-strategy option
  is renamed `linesearch` → `globalization` (the values include
  `:trustregion` and `:tron`, which are not line searches). The
  `AugLagOptions.sub_options` named tuple forwards `globalization` now.
- **Breaking (unreleased branch)**: `sub_approx` is renamed `curvature` and
  subsumes the old `newton` flag — the curvature models are now
  `curvature ∈ (:lbfgs, :sr1, :newton)` (`AugLagOptions(newton = true)`
  becomes `AugLagOptions(curvature = :newton)`). The Greek option names
  are ASCII-ified: `ρ_init/ρ_max/ρ_incr/ρ_decr/auto_ρ_init` →
  `rho_init/rho_max/rho_incr/rho_decr/auto_rho_init`, `η₀` → `eta0`,
  `ω₀` → `omega0`, `elastic_μ` → `elastic_mu`, and the
  `QuasiNewtonAlg` trust-region radius `Δ0` → `delta0`.

- **Defaults updated from the Rosetta measurements**: `outer_schedule = :bm`
  and `sub_tol_schedule = :eta2` are now the defaults (16% faster each on
  Rosetta case5 with the same or better feasibility; `:bm` also 13% faster
  on the TopOpt single-level solve). The box barrier is automatic by
  subproblem order (`box_barrier = missing`): on with `curvature = :newton`, off
  with the first-order subproblems. Confirmed on the TopOpt local-stress
  benchmarks (first-order, PR #356 subproblem budget): 14% faster on the
  single-level p=3 solve (11 vs 13 outer iterations) and 4% faster with a
  lighter feasible design on the SIMP continuation, no feasibility
  regressions.

- **Nonmonotone watchdog (opt-in)**: `AugLagOptions(watchdog = true)`
  adds a Birgin–Martínez-style nonmonotone acceptance test to the outer
  loop — a step is accepted only when its AL merit does not exceed the
  recursive-mean reference `f̄ ← (1−τ)f̄ + τ·merit` (`watchdog_tau`, default
  0.5; the reference starts at the first accepted merit and is frozen on
  rejected steps so persistent failures escalate the penalty), with an
  optional relative slack `merit ≤ (1 + watchdog_tol)·f̄` (`watchdog_tol`,
  default 0.0). Combined with `filter = true`, both tests must pass.
  Measured mixed: inert on the TopOpt local-stress problem (bit-for-bit
  identical trajectories) and a wash on Rosetta case5, but on Rosetta
  cases 14/118 it hurts — with merit-increasing subproblem steps the
  watchdog rejects them outright and the iterate can freeze (case118 ends
  at the start point, f = 0, viol 23 vs 0.62 for the default). A positive
  `watchdog_tol` relaxes the rejection and is the intended setting for
  such problems. Kept as an opt-in safety net.
- **Outer tolerance schedule (opt-in)**: `AugLagOptions(outer_schedule = :bm)`
  switches the feasibility/accuracy targets to the Birgin–Martínez
  complexity scaling (both tighten with the full `1/ρ` factor on good steps
  and reset with `1/ρ` on escalation) instead of the default gentler
  `ρ^-0.9` / `ρ^-0.1` exponents. Measured consistently faster: on the
  TopOpt local-stress single-level `p = 3` solve it converges in 11 vs 13
  outer iterations (312 vs 360 s) at a slightly worse final accuracy, and
  on Rosetta AC-OPF case5 it is 16% faster with marginally better
  feasibility. Also re-measured on Rosetta case5 (300 outer iterations):
  `sub_tol_schedule = :eta2` is 16% faster with the identical trajectory,
  and `penalty = :pbm` / `:mixed` make better feasibility progress than
  `:quadratic` (viol 0.334 vs 0.369) at ~8% less time — the opposite of
  the TopOpt measurement, so `:quadratic` stays the default.
- **Newton subproblem solves (experimental, opt-in)**:
  `AugLagOptions(curvature = :newton)` uses exact Hessians for the AL
  subproblems — from the MathOptInterface evaluator for JuMP-converted
  models (with a stale-Cholesky-preconditioned Steihaug CG), from AD
  otherwise. Measured on Rosetta AC-OPF case5 with `rho_init = 100`:
  converges to `viol = 1e-6` in 86 s while the L-BFGS subproblem solves
  stall at viol ≈ 0.34 after 268 s; with the automatic box barrier it
  converges in 13 outer iterations (10.9 s warm) to the exact optimum at
  an O(μ)-floor bound bias.
- **Adaptive PBM barrier (opt-in)**: `AugLagOptions(pbm_adaptive = true)`
  tracks each inequality group's barrier coefficient to the complementarity
  of its implied duals instead of the global penalty. Measured on Rosetta
  AC-OPF case5: ~10% faster (256 vs 284 s) at the same feasibility.
- **Restoration phase (opt-in)**: `AugLagOptions(restore = true)` recovers
  from subproblem blow-ups with a feasibility-only solve (squared
  violations of the relaxed constraints over the box) instead of reverting
  to the previous point — validated by a NaN-returning subsolver test
  (recovers the feasible point [0.5, 0.5] instead of staying at x0).
- **Box-barrier subproblems (experimental, opt-in)**:
  `AugLagOptions(box_barrier = true)` barrierifies the subproblem's box
  bounds Ipopt-style (log barrier with a C² extension, κ-driven μ
  adaptation, bound-push initialization), with the hard box still projected
  (fraction-to-boundary at τ = 1). Measured on Rosetta AC-OPF case5 with
  `curvature = :newton` and `rho_init = 100`: converges in 12 outer iterations (43.8 s)
  to f = 17553 (the true optimum) at viol 3.5e-8. Second-order only: with
  the first-order (L-BFGS) subproblems it degrades feasibility on case5
  (viol 0.66 vs 0.40). The Steihaug CG also gained inertia-driven
  regularization (τ grows faster after negative-curvature rejections;
  neutral on the measured benchmarks, kept as the Ipopt-correct mechanism).
- **SQSDP (experimental, first cut)**: `SQSDPAlg` / `SQSDPOptions`
  implement sequential quadratic semidefinite programming (Freund-Jarre-
  Vogelbusch style): at each iterate the constraints are linearized (the
  semidefinite one becomes an LMI) and a QP-with-LMI subproblem is solved
  for the step by an injected sub-solver (e.g. `BarrierAlg(IpoptAlg())`),
  globalized by the AL merit with L-BFGS curvature. First-cut verdicts:
  on the BMI it finds the most stable closed loop of all the sd algorithms
  (λ = −1.14 vs −0.82/−1.07) but is slow (barrier-Ipopt subproblems), and
  on the covariance MLE it lands short (f = 56.1 vs f* = 28.7) — exact
  second-order curvature and a faster LSDP sub-solver are the known next
  steps.
- **Two-level semi-proximal inner loop (experimental, opt-in)**:
  `AugLagOptions(two_level = true)` alternates the sd multiplier projection
  with prox-augmented subproblem solves (SDPNAL+-style sPADMM adapted to
  nonlinear `P(x)`). Measured: 12× faster on the N=8 covariance MLE (1.9 vs
  23.2 s, same 1.3e-8 accuracy) and 2.7× faster on the BMI (5.7 vs 15.6 s,
  more stable closed loop).
- **Logdet sd kernels (experimental)**: `penalty = :logdet` and
  `:mixed_logdet` replace the projected quadratic semidefinite penalty with
  the logdet nonlinear rescaling `φ(P) = -(1/ρ) logdet P + ⟨Y, P⟩`
  (Polyak–Teboulle / Kočvara–Stingl modified barrier for the PSD cone,
  clean-room), with the multiplier update `Y ← Π_{S₊ⁿ}(P⁻¹/ρ)` and a
  projected-quadratic fallback outside the cone. Measured on the N=8
  covariance MLE: both converge but land short of the projected quadratic
  (|f-f*| = 0.39 vs 1.3e-8) — the barrier fixed point needs
  complementarity-aware multiplier updates; kept experimental until that
  is added.

## 0.2.10

- Fixed the elastic-mode Anderson acceleration residual: the slack is now included (`ρw(c − s)`), matching the fixed-point map actually being iterated.
## 0.2.9

- Removed a stray planning note from the `AugLagOptions` docstring;
  restructured NEWS.md (newest-first order, no `v` prefixes, added the
  missing 0.1.5-0.2.1 entries).

## 0.2.8

- **100% test coverage** of src/ and the SDP extension: regression tests
  for every option branch (scaling, h0, sub_tol_schedule, selective,
  safeguard, two_way, levenberg, tron_cauchy, elastic + PBM/mixed
  inequality terms, LS-multiplier CG sweeps, TRON negative curvature,
  Cauchy refine fallback and path search, damped-secant guards, structured
  and dense BFGS solve/update paths), plus unit tests for the extracted
  penalty-update helpers.
- **StructuredBFGSApprox seeds the identity** on its structural pattern:
  the previous zero start made every secant update skip (s^T Bs <= 0), so
  the approximation silently degraded to the steepest-descent fallback.
- Removed two unreachable branches (the DenseBFGS solve! Cholesky
  fallback - the update keeps the matrix positive definite - and a dead
  case in the TRON quadratic root helper).

## 0.2.7



- **`h0` default reverts to `:penalty`** (the diagonal `1/(1+ρ)` scaling).
  Re-measured on 0.2.6: `:identity` is 12% slower on single-shot TopOpt
  solves for a ~1% better objective and better PGLib case5 feasibility,
  and on a warm 6-step SIMP continuation it is 9% faster but loses 21% of
  the final objective with a 3x larger stress violation - by the
  faster-only-if-almost-same-accuracy rule, `:penalty` wins both settings
  once the continuation is included.

## 0.2.6



- **L-BFGS base operator defaults to the identity** (`h0 = :identity`; the
  penalty-scaled diagonal remains as `h0 = :penalty`). Re-measured on the
  fixed 0.2.5 code: slightly better TopOpt stress objective (-0.23808 vs
  -0.23559) and better PGLib case5 feasibility (gap 0.10 vs 0.12,
  violation 1.02 vs 1.14) at +12% TopOpt wall-clock.
- **New toggles for the accepted defaults** (each default `true` unless
  noted): `filter`, `stall_retry`, `selective`, `safeguard`, `two_way`,
  `levenberg`, plus `sub_tol_schedule = :fixed | :eta2` and
  `tron_cauchy = false` (Lin-More Cauchy fallback for the TRON option).
  Re-measured: removing the selective weights costs ~79% TopOpt
  wall-clock, removing the safeguard degrades PGLib case14 (gap 0.19 ->
  0.28), removing the Levenberg damping trades speed for quality; the
  filter, stall retry and the two-way rho decrease are inert on the
  measured benchmarks (kept as safety nets).
- `sub_options` now forwards `reset`, `nonmonotone` and `soc` to the
  built-in sub-solver.

## 0.2.5



- Fix the elastic-mode filter backtrack: it evaluated the constraint
  violation on the slack-augmented vector, crashing any model whose
  constraint functions index the variable vector by position
  (e.g. the TopOpt SAND tutorial model).

## 0.2.4



- **Label-based relaxation selection:** `RelaxConstraintSelection(labels = ...)`
  relaxes every constraint whose `flags` contain one of the given labels
  (set with `add_ineq_constraint!(m, f, s; flags = [:relax])`), independently
  of the kind selection.
- Fixes: L-BFGS direct Hessian matvec, label normalization, Anderson residual
  blending, first-iteration failure fallback, dense-BFGS nonpositive sᵀBs
  guard, early gradient-convergence stop in the sub-solver driver.

## 0.2.3



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
  structure mismatch falls back to a cold start. Pays off when the parameter
  changes between solves are small: on the TopOpt local-stress benchmark a
  fine 6-step SIMP schedule ended 21% faster and with a better design than
  cold restarts, while a coarse 3-step schedule traded a better design for
  comparable-or-higher wall-clock.
- **Warm-started L-BFGS across AL outer iterations** (`memory=50`,
  `reset=false`), penalty-scaled diagonal preconditioner, Fletcher-Leyffer
  filter globalization, Anderson acceleration (depth 2), Levenberg damping,
  stall retry, selective penalty weights, safeguarded multipliers, elastic
  mode (opt-in).
- NonconvexIpopt compat widened to `0.7, 0.8`.

## 0.2.2



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

## 0.2.1


- **Block semidefinite constraints** in the projected augmented
  Lagrangian: vector-of-matrices `BlockSDConstraint`s are handled with
  per-block projections and multipliers.

## 0.2.0


- NonconvexCore 2 with strict solver capability declarations.
- Documentation and test coverage of all the ALM options, penalty modes and
  quasi-Newton variants; tests migrated to TestItems with JET, Aqua and
  formatting quality gates.

## 0.1.6


- Widened the compat bounds and adapted the code to Optim 2 and
  IntervalArithmetic 1 APIs.

## 0.1.5


- Fixed AugLag2 equality handling, empty residuals and iteration
  counters, with regression tests for equality rejection and unconstrained
  models.

## 0.1.4



- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.3 and made the package uninstallable on
  Julia 1.10/1.11 (users silently resolved to 0.1.2).
- Requires Julia 1.10+.
