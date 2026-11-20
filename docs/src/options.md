## Options

[`AugLagOptions`](@ref) controls the outer augmented Lagrangian loop:

| Option | Default | Meaning |
|---|---|---|
| `sub_options` | `nothing` | options for the sub-solver (or the built-in `QuasiNewtonAlg`) |
| `relaxed` | `RelaxConstraintSelection()` | which constraint kinds to relax (see below) |
| `penalty` | `:quadratic` | `:quadratic` (classical ALM), `:pbm` (penalty-barrier-multiplier for inequalities) or `:mixed` (PBM for inequalities, quadratic for equalities) |
| `ρ_init` | `10.0` | initial penalty weight |
| `ρ_max` | `1e10` | penalty cap (infeasibility detection) |
| `ρ_incr` | `10.0` | penalty increase factor on insufficient progress |
| `ρ_decr` | `10.0` | penalty decrease factor on successful multiplier steps (two-way adaptation) |
| `max_iter` | `1000` | maximum outer iterations |
| `ctol` | `1e-6` | feasibility tolerance |
| `η₀` | `0.5` | initial feasibility target of the subproblems |
| `ω₀` | `1.0` | initial optimality target of the subproblems |
| `weights` | `nothing` | per-constraint-group penalty weights (default: all ones) |
| `lsq_multipliers` | `true` | initialize equality multipliers from the least-squares stationarity residual (matrix-free CGLS on `Jv`/`Jᵀv` products only) |
| `scaling` | `true` | normalize relaxed constraints by their gradient magnitudes and the objective by its initial magnitude |
| `memory` | `50` | L-BFGS memory of the built-in sub-solver (one warm-started history across outer iterations) |
| `elastic` | `false` | elastic ALM: relax equalities through slack variables |
| `elastic_μ` | `1e3` | slack penalty coefficient of the elastic mode |
| `warm_continue` | `false` | reuse the previous run's iterate, rescaled multipliers, weights and curvature when `optimize!` is called again on the same workspace |
| `keep_all` | `false` | keep the trajectory of outer iterates in the result |

### Penalty modes

The three `penalty` modes differ in how the relaxed *inequality* constraints
enter the augmented Lagrangian objective; relaxed equalities always use the
quadratic term.

```julia
# Classical quadratic ALM: λᵀmax(g,0) + (ρ/2)‖max(g,0)‖²
options = AugLagOptions(penalty = :quadratic)

# PBM (Ben-Tal & Zibulevsky): shifted logarithmic barrier on the feasible
# side, quadratic fallback on the infeasible side
options = AugLagOptions(penalty = :pbm)

# Mixed: PBM for inequalities, quadratic for equalities
options = AugLagOptions(penalty = :mixed)
```

### Relaxation selection

[`RelaxConstraintSelection`](@ref) selects which constraints are relaxed
into the augmented Lagrangian. Each entry is `true` (relax all of that
kind), `false` (keep all hard) or a vector of constraint indices (relax
those only). Box constraints are always handled by the sub-solver.

```julia
# relax all equalities and inequalities
RelaxConstraintSelection()

# keep equalities hard, relax everything else
RelaxConstraintSelection(eq = false)

# relax only the first inequality constraint
RelaxConstraintSelection(eq = false, ineq = [1], sd = false)
```

When `AugLagAlg()` is used without a sub-solver, every constraint must be
relaxed: the built-in `QuasiNewtonAlg` only handles smooth box-constrained
subproblems.
