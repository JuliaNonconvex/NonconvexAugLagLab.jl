# State-of-the-art augmented Lagrangian meta-algorithm.
#
# `AugLagAlg` handles equality, inequality, box and (via the NonconvexSemidefinite
# extension) semidefinite constraints. Constraints selected for relaxation
# (`RelaxConstraintSelection`) are encoded in the augmented Lagrangian
# objective; the rest stay hard for the sub-solver. The outer loop implements
# the practical augmented Lagrangian of Birgin & Martínez: feasibility-driven
# adaptive penalty, tolerance sequences η/ω, multiplier safeguards, two-way
# penalty updates (Curtis et al.) and infeasibility detection. A clean-room
# penalty-barrier-multiplier (PBM) mode follows Ben-Tal & Zibulevsky.
#
# The hardening stack (defaults on, benchmark-gated):
# - Jacobian-free constraint scaling (Curtis-Jiang-Robinson): each relaxed
#   constraint is normalized by its gradient magnitude at x0 and the
#   objective by its initial magnitude, so one penalty balances all groups.
# - Matrix-free CGLS least-squares multiplier initialization (Jv/Jᵀv only).
# - Safeguarded multipliers (clamped) and selective penalty weights that
#   escalate only the most-violated constraint groups.
# - Fletcher-Leyffer filter globalization of the outer loop.
# - Anderson acceleration (depth 2) on the multiplier fixed-point map.
# - One warm-started L-BFGS history (memory 50) across the outer loop with
#   a penalty-scaled base operator H0 = 1/(1+ρ).
# - Levenberg damping and a stall retry for the subproblems.
# Opt-in: TRON projected-CG subproblem solves (`globalization = :tron`), the
# elastic ALM (`elastic = true`), and warm continuation across solves on
# one workspace (`warm_continue = true`). See docs/theory for the details.

"""
    RelaxConstraintSelection(; eq = true, ineq = true, sd = true, sets = true,
                              labels = nothing)

Selects which constraints are relaxed into the augmented Lagrangian. Each
kind entry is `true` (relax all of that kind), `false` (keep all hard) or a
vector of constraint indices (relax those only). In addition, `labels`
(a `Symbol` or a collection of `Symbol`s) relaxes every constraint whose
`flags` (set with `add_ineq_constraint!(m, f, s; flags = [:relax])` and
friends) contain one of the given labels — independently of the kind
selection. Box constraints are always handled by the sub-solver.

`sets` selects the function-in-set constraints ([`FunctionInSet`](@ref)).
Relaxing one splits its components into the equivalent classical
constraints (equality components as equalities, order/simplex/chance
components as inequalities, `PSDCone` components as semidefinite
constraints handled by the projected augmented Lagrangian — which requires
NonconvexSemidefinite to be loaded); the split components are then always
relaxed.

```julia
# relax every constraint carrying the :relax flag, keep the rest hard
RelaxConstraintSelection(eq = false, ineq = false, sd = false, sets = false, labels = [:relax])

# all equalities, plus anything flagged :relax
RelaxConstraintSelection(labels = :relax)
```
"""
struct RelaxConstraintSelection
    eq::Union{Bool,AbstractVector{Int}}
    ineq::Union{Bool,AbstractVector{Int}}
    sd::Union{Bool,AbstractVector{Int}}
    sets::Union{Bool,AbstractVector{Int}}
    labels::Union{Nothing,Set{Symbol}}
end
function RelaxConstraintSelection(; eq=true, ineq=true, sd=true, sets=true, labels=nothing)
    lbls = if labels === nothing
        nothing
    elseif labels isa Union{AbstractVector,Tuple,Base.AbstractSet}
        Set(Symbol.(labels))
    else
        # a single label (Symbol or String): Symbol.(x) would broadcast
        # (and a bare Set(x) cannot iterate a scalar), so wrap it first
        Set([Symbol(labels)])
    end
    return RelaxConstraintSelection(eq, ineq, sd, sets, lbls)
end

function _relaxed_indices(sel::Bool, n::Int)
    return sel ? collect(1:n) : Int[]
end
function _relaxed_indices(sel::AbstractVector{Int}, n::Int)
    return sel
end
function _relaxed_indices(sel, n::Int)
    return throw(
        ArgumentError(
            "Relaxation selection entries must be `true`, `false` or a vector of " *
            "constraint indices; got $(typeof(sel)).",
        ),
    )
end

# The flags (labels) of a constraint, or the empty set for constraint types
# that do not carry flags (e.g. JuMP-converted DictModel constraints).
function _flags(c)
    return hasfield(typeof(c), :flags) ? c.flags : Set{Symbol}()
end

"""
    SecondOrderSelection(; eq = true, ineq = true, labels = nothing)

Selects which relaxed constraint groups contribute their exact
multiplier-weighted curvature `μᵀ∇²c` to the hybrid Hessian model
(`curvature = :hybrid`). Each kind entry is `true` (all groups of that
kind), `false` (none) or a vector of group indices (those only);
`labels` selects every constraint whose `flags` contain one of the given
labels. Untagged groups keep only the Gauss-Newton penalty term and
their multiplier curvature is absorbed by the quasi-Newton remainder.
Only JuMP-converted blocks support the exact term (MathOptInterface
Lagrangian-Hessian triplets).
"""
struct SecondOrderSelection
    eq::Union{Bool,AbstractVector{Int}}
    ineq::Union{Bool,AbstractVector{Int}}
    labels::Union{Nothing,Set{Symbol}}
end
function SecondOrderSelection(; eq=true, ineq=true, labels=nothing)
    lbls = if labels === nothing
        nothing
    elseif labels isa Union{AbstractVector,Tuple,Base.AbstractSet}
        Set(Symbol.(labels))
    else
        Set([Symbol(labels)])
    end
    return SecondOrderSelection(eq, ineq, lbls)
end

function _so_selected(sel::Bool, k, n)
    return sel
end
function _so_selected(sel::AbstractVector{Int}, k, n)
    return k in sel
end
_so_selected(sel, k, n) = false
function _so_labeled(c, labels)
    return labels !== nothing && !isempty(intersect(_flags(c), labels))
end

# Per-group second-order flags of the relaxed constraint groups.
function _second_order_flags(sel, model, eq_rel, ineq_rel)
    so_eq = falses(length(eq_rel))
    so_ineq = falses(length(ineq_rel))
    sel === nothing && return so_eq, so_ineq
    for (k, i) in enumerate(eq_rel)
        so_eq[k] =
            _so_selected(sel.eq, k, length(eq_rel)) ||
            _so_labeled(model.eq_constraints.fs[i], sel.labels)
    end
    for (k, i) in enumerate(ineq_rel)
        so_ineq[k] =
            _so_selected(sel.ineq, k, length(ineq_rel)) ||
            _so_labeled(model.ineq_constraints.fs[i], sel.labels)
    end
    return so_eq, so_ineq
end

# Relaxation indices of a constraint list: the kind-based selection plus
# every constraint carrying one of the selected labels. NB: the count is
# `length(fs.fs)` — `length(::VectorOfFunctions)` is its dimension
# (`Base.length(::AbstractFunction) = getdim`), not the number of constraints.
function _relaxed_indices(sel, fs::VectorOfFunctions, labels::Union{Nothing,Set{Symbol}})
    idx = _relaxed_indices(sel, length(fs.fs))
    labels === nothing && return idx
    # copy before push!/sort!: `sel` may be the user's array (must not be
    # mutated) or an AbstractRange (cannot be pushed to)
    idx = collect(idx)
    for (i, c) in enumerate(fs.fs)
        if !(i in idx) && !isempty(intersect(_flags(c), labels))
            push!(idx, i)
        end
    end
    return sort!(idx)
end

"""
    AugLagAlg(sub_alg = nothing)

Augmented Lagrangian meta-algorithm. `sub_alg` solves the subproblems after the
relaxed constraints are folded into the objective; `nothing` selects the
built-in [`QuasiNewtonAlg`](@ref), which requires every constraint to be
relaxed. Relaxing semidefinite constraints additionally requires
NonconvexSemidefinite to be loaded (its extension provides the cone
projections).
"""
struct AugLagAlg{S<:Union{Nothing,AbstractOptimizer}} <: AbstractOptimizer
    sub_alg::S
end
AugLagAlg() = AugLagAlg(nothing)
AugLagAlg(sub_alg::AbstractOptimizer) = AugLagAlg{typeof(sub_alg)}(sub_alg)

function _optimize_precheck(model::AbstractModel, optimizer::AugLagAlg, args...; kwargs...)
    return nothing
end

function NonconvexCore.solver_capabilities(alg::AugLagAlg)
    if alg.sub_alg === nothing
        # the built-in QuasiNewtonAlg handles box-constrained subproblems
        return ConstraintCapabilities(;
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            linear_sd=true,
            nonlinear_sd=true,
            sets_eq=true,
            sets_ineq=true,
            sets_sd=true,
            sets_simplex=true,
            sets_product=true,
            sets_chance=true,
            variable_sets_sd=true,
            variable_sets_simplex=true,
            box=true,
            continuous=true,
            binary=false,
            integer=false,
            infinite_bounds=true,
        )
    end
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(
        sub;
        linear_eq=true,
        nonlinear_eq=true,
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=true,
        nonlinear_sd=true,
        sets_eq=true,
        sets_ineq=true,
        sets_sd=true,
        sets_simplex=true,
        sets_product=true,
        sets_chance=true,
        variable_sets_sd=true,
        variable_sets_simplex=true,
    )
end

"""
    AugLagOptions(; kwargs...)

- `sub_options`: options for the sub-solver. For the built-in
  `QuasiNewtonAlg` this is a named tuple of its keyword arguments (default:
  `(; max_iter = 800, tol = 1e-8, globalization = :trustregion)` — trust-region
  Steihaug–Toint steps are substantially more robust than backtracking on
  the nonconvex augmented Lagrangian subproblems); for a custom `sub_alg`
  it is forwarded as its `options`.
- `relaxed`: a [`RelaxConstraintSelection`](@ref) (default: relax everything).
- `penalty`: `:quadratic` (classical ALM), `:pbm` (penalty-barrier-multiplier
  for inequalities, quadratic for equalities), `:mixed` (PBM for
  inequalities, projected quadratic for semidefinite), `:logdet` (logdet
  nonlinear rescaling for semidefinite, quadratic elsewhere) or
  `:mixed_logdet` (PBM for inequalities, logdet for semidefinite). The
  logdet kernels are experimental: measured on the covariance MLE they
  converge but land short of the projected quadratic.
- `auto_rho_init`: (opt-in) calibrate the initial penalty from the problem at
  x0: `ρ = clamp(‖∇f(x0)‖ / max(‖c̃(x0)‖, ε), 1, 1e6)` on the scaled
  constraints — the penalty gradient then balances the objective gradient,
  avoiding the measured escalation waste (the default `rho_init = 10` needs
  ~130 outer iterations of escalation on AC-OPF). Measured: wins Rosetta
  AC-OPF case5 (viol 0.34 vs 0.60 at 200 outer iterations) but loses the
  TopOpt local-stress problem badly (near-feasible starts give a tiny
  scaled violation and an over-large penalty: viol 0.009 vs 5.7e-6, 7×
  slower) — opt-in for infeasible-start problems, off by default.
- `rho_init`, `rho_max`, `rho_incr`, `rho_decr`: initial/maximum penalty and the
  increase/decrease factors (two-way adaptation).
- `max_iter`: maximum outer iterations.
- `ctol`: feasibility tolerance.
- `eta0`, `omega0`: initial feasibility/optimality targets of the subproblems.
- `weights`: per-constraint-group penalty weights (default: all ones).
- `lsq_multipliers`: initialize equality multipliers from the least-squares
  stationarity residual (default `true`).
- `prox`: (opt-in) proximal regularization of the AL subproblems
  (Rockafellar): adds `τ/2‖x − x̄‖²` to the subproblem objective, anchored
  at the outer iterate — conditions the ill-conditioned subproblems of the
  escalated penalty. `0` (default) disables it. Composes with every penalty
  mode (`:bm`, `:pbm`, `:mixed`, `:logdet`, `:mixed_logdet`) and with
  `curvature = :sr1`; with `curvature = :newton`, the exact-Hessian subproblems add the
  τI term (the prox term's true curvature). Only the `two_level` inner loop
  rebuilds from the un-proxed objective, so its inner anchor replaces the
  outer prox. Measured: 10% faster with slightly better feasibility on
  Rosetta AC-OPF case5 (151.7 vs 169.2 s, viol 0.584 vs 0.599), and on
  TopOpt local-stress 32% faster and more feasible on the single-level
  solve but converging to a heavier design (obj 0.181 vs 0.094 — the
  anchor bias), while the 3-stage SIMP continuation flips that trade:
  2.7× faster (204 vs 560 s) *and* a lighter final design (obj 0.186 vs
  0.187) at viol 5.3e-9 — opt-in. Combinations measured on Rosetta case5:
  with `penalty = :pbm`, prox is 13% faster and lands on a lower f but
  less feasible (159.5 s, viol 0.524 vs 183.4 s, 0.399 — no dominant win);
  with `curvature = :sr1`, prox is 27% faster but far from the optimum
  (57.6 s, f = 14187, viol 0.158 vs 78.9 s, f = 17576, viol 0.016) — the
  anchor counteracts SR1's curvature exploration, so the combination is
  not recommended.
- `memory`: L-BFGS memory of the built-in subproblem solver (default `50`;
  the curvature history is reused across outer iterations).
- `curvature`: the curvature approximation of the built-in subproblem
  solver, `:lbfgs` (default), `:sr1` (limited-memory symmetric rank-one —
  captures indefiniteness, intended for the trust-region subproblems),
  `:newton` (experimental, opt-in: exact-Hessian Newton trust-region
  solves — MathOptInterface evaluator Hessians for JuMP-converted models,
  AD otherwise, with the stale-Cholesky preconditioned Steihaug CG; the
  box-barrier mode auto-pairs with it) or `:hybrid` (experimental,
  opt-in: the trust-region Hessian model sums the exact Gauss-Newton
  penalty terms `ρw·JᵀJ` of the relaxed constraint groups — matrix-free
  through one forward finite difference and one reverse pass each, with
  the inequality rows restricted to the active set frozen at the
  subproblem iterate — plus an L-BFGS model of the remainder
  (`λ`-weighted terms; the composite's quasi-Newton part covers the
  remainder with the structured-secant update (Dennis–Gay–Welsch: the
  exact terms' curvature is subtracted from the secant pairs, so the
  QN models only the remainder), and the identity base operator is used
  since the penalty scale is already exact in the terms). Experimental;
  measured on the TopOpt local-stress single-level solve: the matvec
  cost dominates (a forward finite difference + a reverse pass per CG
  iteration against full-FEA stress constraints) — no completion within
  20 min vs the 4.5 min L-BFGS baseline; on Rosetta case5 the
  JuMP-converted blocks use the exact sparse evaluator Jacobian (two
  sparse matvecs per CG iteration) and, with the `second_order` labels,
  converge to machine-precision feasibility in 29 outer iterations /
  34.3 s (see `second_order` below) — faster than SR1 (80 s) and Newton
  (43.8 s), at a 1.4% higher objective (a different feasible local
  solution). The remainder-only L-BFGS inverse was also measured as a
  Steihaug CG preconditioner and is deliberately not wired (it stalled
  the solve on case5). Opt-in.
- `curvature = :split` (experimental, opt-in): the split-secant model
  for problems without exact Hessians — the penalty part
  `φ = ρ/2‖c‖² + ρ/2‖g⁺‖²` and the remainder `ψ = f + yᵀc + λᵀg⁺` get
  separate L-BFGS histories: the penalty pair `(s, ∇φ(x⁺) − ∇φ(x))` is
  the exact directional second derivative of the dominant penalty
  curvature from gradient differences alone (two reverse passes of the
  scalar penalty function per accepted step, no Hessian-vector
  products), the remainder receives the structured-secant corrected
  pair `(s, y − Δ∇φ)` (Dennis–Gay–Welsch in gradient-difference form),
  the CG model is the sum of the two two-loop recursions and the
  penalty model's inverse is the Steihaug preconditioner. Quadratic
  penalty mode, no elastic mode, any model (no JuMP requirement).
  Measured on the TopOpt local-stress single-level solve: no completion
  within 40 min vs the 4.5 min L-BFGS baseline — the two penalty-part
  reverse passes per accepted step (full FEA adjoints) dominate; the
  design targets problems with cheap constraint evaluations.
- `curvature = :fdgrad` (experimental, opt-in): inexact-Newton curvature
  from one-sided finite differences of the Zygote-computed AL gradient
  (`Hv = (g(x+εv) − g(x))/ε`, reusing the driver's current gradient so
  each Hessian-vector product costs one extra gradient evaluation;
  Zygote cannot be nested and ForwardDiff cannot cross its tape); pair
  with `cg_maxiter` (via `sub_options`) for a small product budget.
  Measured on the TopOpt local-stress single-level solve: no completion
  within 15 min at `cg_maxiter = 5` vs the 4.5 min L-BFGS baseline —
  the per-CG-iteration gradient evaluations dominate on FEA-heavy
  constraints; a single CG iteration with the L-BFGS preconditioner
  (`cg_maxiter = 1`, `precond = true`, plain `:lbfgs`) also loses (no
  completion within 28 min: steepest-descent-quality steps).
  Measured: `:sr1` is a massive- `second_order`: (opt-in, with `curvature = :hybrid`) a
  [`SecondOrderSelection`](@ref) selecting which relaxed constraint
  groups contribute their exact multiplier-weighted curvature `μᵀ∇²c`
  (MathOptInterface Lagrangian-Hessian triplets, one sparse assembly
  per subproblem iterate) on top of the Gauss-Newton term; untagged
  groups keep the Gauss-Newton term only, with their multiplier
  curvature absorbed by the L-BFGS remainder. JuMP-converted blocks
  only. `nothing` (default) disables the exact terms. Measured on
  Rosetta case5: labeling all groups (with the structured-secant
  composite update) converges to machine-precision feasibility in
  29 outer iterations / 34.3 s (viol 9.3e-14, f = 17808 — 1.4% above
  the 17553 optimum Newton finds, a different feasible local solution)
  vs 80 s for SR1 and 43.8 s for Newton.
- `inexact_cg`: (opt-in) pass the subproblem tolerance to the built-in
  trust-region solver as a relative Steihaug CG residual tolerance
  (Dembo–Eisenstat–Steihaug inexact Newton: the CG stops once
  `‖r‖ ≤ sub_tol·‖r₀‖`, so loose subproblems run few Hessian-vector
  products and the budget tightens to zero as the outer loop
  converges). `false` (default) keeps the exact CG. Measured on Rosetta
  case5 (hybrid + all labels): bit-for-bit the same trajectory at 3.6%
  less wall-clock (251.9 vs 261.4 s) — a small win, off by default
  pending the TopOpt characterization.
- `sub_options.cg_maxiter`, `sub_options.precond` (for the built-in
  quasi-Newton solver): cap the Steihaug CG iterations and enable the
  L-BFGS two-loop as the CG preconditioner (off by default — the
  preconditioner changes the steps). Measured on the TopOpt stress
  solve: `cg_maxiter = 1` + `precond = true` takes
  steepest-descent-quality steps (no completion within 28 min vs the
  4.5 min baseline); the split model's penalty-inverse preconditioner
  also activates through `precond = true`.
  Rosetta AC-OPF case5 win (80 vs 219 s, viol 0.016 vs 0.60, f near the
  true optimum) but slower on the TopOpt local-stress problem (no
  completion within 31 min vs ~6 min for L-BFGS) — opt-in; `:newton` with
  `box_barrier` solves case5 to the true optimum (43.8 s, f = 17553,
  viol 3.5e-8) — second-order only, opt-in.
- `elastic`: relax the equalities through slack variables,
  `yᵀ(c−s) + ρ/2‖c−s‖² + μΣ√(sᵢ²+ε²)`, the elastic mode of the augmented
  Lagrangian (more robust to infeasible subproblems; default `false`).
- `elastic_mu`: the slack penalty coefficient of the elastic mode.
- `warm_continue`: reuse the previous run's final iterate, multipliers,
  per-group weights and quasi-Newton curvature when `optimize!` is called
  again on the same workspace (e.g. across the stages of a SIMP
  continuation, where the model or the hyper-parameters change between
  solves); the multipliers are rescaled to the new constraint scales, while
  `ρ`, `η`, `ω`, the filter and the Anderson history are re-initialized from
  the new options. Measured on the released 0.2.5 code (TopOpt local-stress
  SIMP continuation): warm continuation is consistently *faster* than cold
  restarts (26% on a coarse 3-step schedule, 10% on a fine 6-step one) and
  keeps the design at the stress threshold (violation 0.0 vs 0.005 on 3
  steps, 0.003 vs 0.032 on 6 steps), while cold restarts end with a lighter
  objective but slightly violate the stress limit. Use it when feasibility
  or runtime dominate the continuation, not for a lower final objective.
- `h0`: base operator of the built-in L-BFGS recursion, `:penalty` (the
  default diagonal `1/(1+ρ)` scaling) or `:identity`. Re-measured on
  0.2.5/0.2.6: `:identity` is 12% slower on single-shot TopOpt solves for
  a 1% better objective and better PGLib case5 feasibility (gap 0.10 vs
  0.12, violation 1.02 vs 1.14), while on a warm 6-step SIMP continuation
  (`warm_continue = true`) it is 9% faster but loses 21% of the final
  objective (0.198 vs 0.163) with a 3× larger stress violation — the
  faster variant is not close enough in accuracy there, so `:penalty`
  stays the default.
- `hard_rho`: the penalty weight (default `1e3`, must be positive) for
  constraints *kept hard* with the built-in sub-solver
  (`AugLagAlg()` with a partially-relaxing `RelaxConstraintSelection`):
  the kept equality/inequality constraints are folded into the
  subproblem objective as the smooth quadratic penalty
  `hard_rho/2·(‖c‖² + ‖g⁺‖²)` (the ALM-for-relaxed / penalty-for-kept
  split) and removed from the submodel so the box-only quasi-Newton
  solver accepts it; the kept constraints' feasibility is traded against
  `hard_rho` (large values drive their violation to zero). External
  sub-solvers receive the kept constraints natively instead. Measured on
  the TopOpt local-stress single-level solve with the stress constraints
  kept hard: converges in one outer iteration (132.7 s) but leaves the
  expected `hard_rho` residual — max solid stress 5.7e-4 over the
  threshold at a 2.2× heavier design; the relaxed ALM is strictly better
  there, this path is for problems with a cheap-hard/expensive-soft
  split.
- `watchdog`, `watchdog_tau`, `watchdog_tol`: Birgin–Martínez-style
  nonmonotone outer acceptance (default `false`). When enabled, a step is
  accepted only if its AL merit does not exceed the recursive-mean reference
  `f̄ ← (1-τ)·f̄ + τ·merit` (τ = `watchdog_tau`; the reference starts at the
  first accepted merit and is left unchanged on rejected steps, so repeated
  failures eventually escalate the penalty), with an optional relative slack
  `merit ≤ (1 + watchdog_tol)·f̄` (default `0.0`). Combined with `filter`,
  both acceptance tests must pass. Opt-in, measured mixed: on the TopOpt
  local-stress problem the trajectories are bit-for-bit identical to the
  default (the subsolver always returns merit-decreasing steps), on Rosetta
  AC-OPF case5 it is a wash (milestones within 1–3 outer iterations), but
  on cases 14/118 it *hurts* — with merit-increasing subproblem steps the
  watchdog rejects them outright and the iterate can freeze (case118 ends
  at the start point, f = 0, viol 23 vs 0.62 for the default). A positive
  `watchdog_tol` relaxes the rejection and is the intended setting for such
  problems. Kept as an opt-in safety net for subsolvers that reliably
  return bad steps.
- `anderson_ineq`: (opt-in) extend the Anderson acceleration (currently
  equality-only) to the inequality multiplier map of the classical
  quadratic penalty: the unprojected update `λ + ρw·g` is mixed with the
  last two iterates (same depth-2 least-squares mixing as the equality
  block) and the mix is re-projected onto the nonnegative orthant
  (ADMM-style Anderson with re-projection). Inactive for the PBM penalty
  modes. Measured on the TopOpt local-stress single-level solve (current
  default config): a wall-clock wash (262.6 vs 264.6 s) with a 4.6%
  lighter feasible design (obj 0.0895 vs 0.0938, reproduced exactly
  across configurations) and a marginally larger final violation
  (6.2e-6 vs 5.7e-6) — no time win, so it stays off by default; the
  opt-in "light-design recipe". With `anderson_depth = 5` the design is
  a further 0.2% lighter and the violation beats the default (obj
  0.0893, viol 5.4e-6) at the same wall-clock.
- `anderson_depth`: the history depth of the Anderson mixing (default
  `2`, must be ≥ 1; `1` disables the mixing). Applies to both the
  equality map and, when `anderson_ineq` is enabled, the inequality map.
- `multiplier_relax`: (opt-in) Eckstein-style adaptive relaxation of the
  multiplier step: when the subproblem stationarity residual at the
  accepted iterate (the projected gradient of the subproblem objective,
  the built-in solver's own stopping metric) exceeds the requested
  subproblem tolerance, the dual increment is scaled down by the
  achieved-accuracy ratio `θ = clamp(sub_tol/‖res‖, 0.1, 1)` — a loose
  inner solve cannot poison the multiplier estimates with a full step.
  Applies to the equality updates and the classical quadratic inequality
  updates (the Anderson residuals are scaled to match); the PBM/SD
  multiplier updates are left unscaled. One extra subproblem-gradient
  evaluation per outer iteration when enabled. Measured on the TopOpt
  local-stress single-level solve: 2.1% slower (309.5 vs 303.1 s) but a
  2.1% lighter feasible design (obj 0.0918 vs 0.0938) with marginally
  better feasibility — a small quality-for-time trade, off by default.
- `spectral_rho`: (opt-in) replace the two-way penalty decrease on good
  steps with a spectral (Barzilai–Borwein) update of the dual step:
  `ρ ← ‖Δc‖²/⟨Δc, Δy⟩` from the last two iterates of the dual map
  (Xu–Figueiredo–Goldstein style), applied only on a credible positive
  correlation, limited to `[ρ/rho_decr, ρ·rho_incr]` and
  `[rho_init/10, rho_max]`. Equality and inequality multipliers
  participate (semidefinite matrices do not); the elastic mode is not
  supported (falls back to the two-way rule there). Escalation on bad
  steps is unchanged. Measured on the TopOpt local-stress single-level
  solve (current default config): 3.0% slower (272.6 vs 264.6 s) and a
  1.3% heavier design (obj 0.0950 vs 0.0938) — a straight loss, off by
  default. (An earlier 6.2%-faster measurement was against a
  fixed-tolerance configuration and does not carry over.)
- `dual_secant`: (opt-in) Broyden secant correction on the equality
  multiplier map (the Birgin–Martínez "improving ultimate convergence"
  device): the inverse secant model of the residual map of `y += ρw·c` is
  refined with the good-Broyden inverse update from the map's secant
  pairs, and the multiplier step becomes the quasi-Newton step
  `y − C·(ρw·c)` (the first step reproduces the plain ascent), safeguarded
  back to the plain ascent when the model step exceeds twice its norm.
  Replaces the plain update and the equality Anderson mixing when
  enabled; inactive for the elastic mode and without relaxed equalities.
  Awaiting the AC-OPF measurement (the TopOpt stress problem has no
  relaxed equalities, so the path is inert there).
- `filter`, `stall_retry`, `selective`, `safeguard`, `two_way`, `levenberg`:
  toggles for the corresponding defaults (all `true`). Re-measured on 0.2.5:
  removing the selective weights costs ~79% TopOpt wall-clock, removing the
  multiplier safeguard degrades PGLib case14 (gap 0.19 → 0.28), removing the
  Levenberg damping trades ~30% TopOpt speed for a worse objective and
  feasibility; the filter, stall retry and the two-way ρ decrease are inert
  on the measured benchmarks (kept as safety nets).
- `sub_tol_schedule`: `:eta` (the default), `:eta2`, `:fixed` (`1e-8`) or
  `:relative`. The default `:eta` schedule loosens the subproblem tolerance as the outer
  feasibility target tightens (clamped to `[1e-6, 1e-2]`): cheap sub-solves
  while the outer loop only needs rough steps, floored at `1e-6` for the
  final accuracy (measured 28% faster than `:eta2` on TopOpt stress with
  4.8x tighter final feasibility). `:eta2` (clamped to `[1e-10, 1e-8]`)
  remains for maximal final accuracy. `:relative` sets the inner tolerance
  proportional to the current constraint violation,
  `clamp(sub_tol_rel * v_prev, 1e-10, 1e-2)` with `sub_tol_rel` (default
  `0.1`, must be in `(0, 1)`): the Eckstein–Silva practical relative
  criterion — the subproblem is solved only to an accuracy comparable to
  the current infeasibility, so effort concentrates where it pays. Measured
  on the TopOpt local-stress single-level solve: bit-for-bit the same
  trajectory as the `:eta` schedule at 3% more wall-clock (290.9 vs
  282.2 s) — no TopOpt win, so `:eta` stays the default.
  The first subproblem is tightened separately by `first_sub_tol` (see
  below).
  The schedule governs every sub-solver:
  external solvers (Ipopt, MMA, NLopt) receive the current tolerance through
  their options via the package extensions (`_set_sub_tol`).
- `first_sub_tol`: the first subproblem is solved to
  `min(sub_tol, first_sub_tol)` (default `1e-6`); the first inner solve
  selects the local basin the whole solve follows (measured on the TopOpt
  local-stress problem: the schedule's loose first solve lands in a 4%
  heavier design's basin), so it is tightened while later subproblems keep
  the cheap schedule tolerance. Measured: the default `:eta` schedule plus
  `first_sub_tol = 1e-6` reproduces the light-design trajectory of a
  fixed-`1e-6` override bit-for-bit at 12.7% less wall-clock (264.6 vs
  303.1 s). Set `first_sub_tol = nothing` for the pre-0.4.1 behavior.
- `first_sub_iters`: how many of the leading subproblems get the
  `first_sub_tol` tightening (default `1`, must be ≥ 1). Measured on the
  TopOpt single-level solve: `2` is bit-for-bit inert (262.0 vs 264.6 s,
  identical trajectory) — the basin is selected by the first solve alone.
- `max_time`: a wall-clock budget in seconds (default `Inf`): no new
  subproblem is started once the budget is exhausted, and the solve
  returns the last completed outer iterate.
- `on_iteration`: (opt-in) a per-iteration callback
  `on_iteration(iter, x, f(x), vnorm)` called after every accepted outer
  iterate (with a copy of the iterate); returning `true` stops the loop
  and reports the just-recorded iterate.
- `analytical_gradient`: (opt-in) assemble the built-in subproblem's
  gradient analytically as `∇f + Σ Jᵀv` — one vector-Jacobian product per
  constraint group through the existing reverse-mode machinery, exposed
  via a ChainRules rrule (the SDP-extension pattern) — instead of
  differentiating the composed AL closure. Quadratic penalty mode only
  (no PBM, elastic or `curvature = :newton`). Measured on the TopOpt
  local-stress single-level solve: bit-for-bit the same trajectory as
  the composed-AD path at 17.8% more wall-clock (311.6 vs 264.6 s) — the
  per-group reverse passes cost more than the single composed pass on
  this one-group problem; off by default, an architecture/robustness
  option for multi-group problems.
- `stop_kkt`, `kkt_tol`: the default convergence test is KKT-complete:
  once `vnorm ≤ ctol`, the solve also requires the box-projected gradient
  of the scaled Lagrangian at the current multipliers to satisfy
  `≤ kkt_tol` (default `1e-3`) and the complementarity residual
  `max |λⱼgⱼ|` over the relaxed inequalities to satisfy `≤ kkt_tol`.
  Costs one Lagrangian-gradient assembly per outer iteration once
  feasible. A feasibility-only solve would terminate at any feasible
  iterate — including the starting point of a feasible-start problem —
  so stationarity is part of the default test; the tolerance is set at
  the residual plateau measured on the TopOpt local-stress single-level
  solve (the multiplier-error scale is ~ρ·viol, so the Lagrangian
  residual stalls around 1e-4-1e-3 on that 1600-constraint problem, where
  the old `1e-6` default never fired within 100 outer iterations). Set
  `stop_kkt = false` for the feasibility-only behavior, or loosen
  `kkt_tol` and bound the run with `max_time` on large problems.
- `outer_schedule`: `:bm` (the default) — the Birgin–Martínez complexity
  scaling where both the feasibility target `η` and the subproblem accuracy
  target `ω` tighten with the full `1/ρ` factor on good steps (and reset
  with `1/ρ` on escalation) — or `:default` (the gentler `ρ^-0.9` /
  `ρ^-0.1` exponents, the pre-0.2.10 behavior). Measured consistently
  faster: on the TopOpt local-stress single-level `p = 3` solve it
  converges in 11 vs 13 outer iterations (312 vs 360 s) at a slightly
  worse final accuracy (viol 5.7e-6 vs 1.2e-6), and on Rosetta AC-OPF
  case5 it is 16% faster (217 vs 258 s) with marginally better feasibility
  (viol 0.366 vs 0.369).
- `scaling`: normalize the relaxed constraints by their gradient magnitudes
  and the objective by its initial magnitude (Curtis–Jiang–Robinson style
  scaling), so the penalty applies to balanced, unit-scaled constraint
  violations (default `true`).
- `two_level`, `two_level_iters`, `two_level_tau`: (experimental, opt-in)
  two-level semi-proximal inner loop for the semidefinite block
  (SDPNAL+-style sPADMM adapted to nonlinear `P(x)`, plan S2 option B):
  after each subproblem solve, alternate the multiplier projection with
  `two_level_iters` prox-augmented subproblem solves
  (`AL(x) + τ/2‖x − x̂‖²`). Requires the built-in sub-solver and relaxed
  semidefinite constraints. Measured: 12× faster on the N=8 covariance MLE
  (1.9 vs 23.2 s, same 1.3e-8 accuracy) and 2.7× faster on the BMI (5.7 vs
  15.6 s, with a more stable closed loop, λ = −1.07 vs −0.82).
- `box_barrier`, `box_barrier_scale`: (experimental, opt-in) Ipopt-style
  barrierification of the subproblem's box bounds: the box enters the
  subproblem objective as a log barrier `−μ Σ(log(x−l) + log(u−x))` with a
  C² extension outside, `μ` adapts by the implied-dual complementarity on
  the near-bound coordinates (Ipopt-style κ rule), the start is pushed off
  the bounds by the barrier thickness (bound-push), and the hard box stays
  projected in the subproblem (fraction-to-boundary at τ = 1) so the
  iterate never leaves the bounds.
  `box_barrier = missing` (the default) is automatic: **on** with
  `curvature = :newton` (measured on Rosetta case5: 43.8 s to f = 17553, viol 3.5e-8)
  and **off** with the first-order subproblems (measured harmful there:
  viol 0.66 vs 0.40).
- `pbm_adaptive`: (opt-in) Ipopt-style barrier adaptation for the PBM
  penalty modes: each inequality group's barrier coefficient tracks the
  complementarity of its implied duals (grow the barrier when the
  complementarity exceeds the target, shrink it when well-complemented)
  instead of following the global penalty. Measured on Rosetta AC-OPF
  case5: ~10% faster (256 vs 284 s) at the same feasibility.
- `restore`: (opt-in) Ipopt-style restoration phase: when the subproblem
  iterate blows up (non-finite or exploding violation) instead of just
  reverting to the previous point, solve a feasibility-only problem
  (`min Σ‖c‖² + Σ‖g⁺‖² + Σ‖viol_sd‖²` over the box, on the same scaled
  constraints as the main phase) with the built-in quasi-Newton driver to
  recover a usable iterate, then continue.
- `restore_stall`: (opt-in, with `restore`) also trigger the restoration
  when feasibility stalls: no improvement of more than `restore_tol` over
  the last `restore_stall` outer iterations. The restored point replaces
  the iterate and the equality multipliers are re-estimated (least
  squares) there; at most 2 restorations per solve. `0` (default) keeps
  the blow-up-only trigger.
- `restore_tol`: the feasibility-improvement threshold for the
  `restore_stall` trigger, in the units of `restore_tol_mode`: `:relative`
  (default `1e-2`: the violation must have dropped by more than 1% over
  the window) or `:absolute` (a drop in the raw scaled violation). The
  main phase is not feasibility-preserving, so when a restoration point is
  strictly more feasible than the final main-phase iterate it is reported
  as the result. Measured on Rosetta AC-OPF case5: `restore = true,
  restore_stall = 5` drives the violation from 0.60 to 1.6e-6 and the
  objective to f = 17858 (the true optimum) in ~300 s (vs the 169.7 s /
  viol 0.60 / f 8921 default) — a decisive opt-in win. On the TopOpt SAND
  tutorial it fires but cannot close the equilibrium gap: the raw merit
  Σ‖Ku−f‖² is a KᵀK least-squares whose SIMP stiffness conditioning
  κ(K)² stalls any first-order restore at resid ~0.08 (only the nested
  solve u = K⁻¹f — the NAND formulation from the SAND literature — or
  second-order steps reach equilibrium there).
- `restore_prox`: (opt-in) Rockafellar anchor weight `τ` on the
  restoration objective: `min Σ‖c‖² + Σ‖g⁺‖² + τ/2‖z − x̄‖²` with `x̄` the
  pre-restoration iterate — the violation merit alone can wander along its
  zero manifold; `τ > 0` keeps the restored point close to the previous
  solution. The anchor is a per-variable scaled pseudo-Huber (log-cosh)
  penalty: each variable is scaled by its box width, and the anchor
  gradient is `τ·tanh(Δᵢ/wᵢ)` — magnitude below `τ` for every variable,
  saturating instead of growing linearly. `0` (default) disables the
  anchor.
- `keep_all`: keep the trajectory of outer iterates in the result.
"""
mutable struct AugLagOptions{S,R,W}
    sub_options::S
    relaxed::R
    penalty::Symbol
    rho_init::Real
    rho_max::Real
    rho_incr::Real
    rho_decr::Real
    auto_rho_init::Bool
    curvature::Symbol
    prox::Real
    max_iter::Int
    ctol::Real
    eta0::Real
    omega0::Real
    weights::W
    lsq_multipliers::Bool
    scaling::Bool
    memory::Int
    elastic::Bool
    elastic_mu::Real
    warm_continue::Bool
    filter::Bool
    watchdog::Bool
    watchdog_tau::Real
    watchdog_tol::Real
    anderson_ineq::Bool
    anderson_depth::Int
    multiplier_relax::Bool
    spectral_rho::Bool
    dual_secant::Bool
    stall_retry::Bool
    selective::Bool
    safeguard::Bool
    two_way::Bool
    levenberg::Bool
    h0::Symbol
    hard_rho::Real
    sub_tol_schedule::Symbol
    sub_tol_rel::Real
    first_sub_tol::Union{Nothing,Real}
    first_sub_iters::Int
    max_time::Real
    on_iteration::Union{Nothing,Function}
    analytical_gradient::Bool
    stop_kkt::Bool
    kkt_tol::Real
    second_order::Union{Nothing,SecondOrderSelection}
    inexact_cg::Bool
    outer_schedule::Symbol

    two_level::Bool
    two_level_iters::Int
    two_level_tau::Real
    box_barrier::Union{Bool,Missing}
    box_barrier_scale::Real
    restore::Bool
    restore_stall::Int
    restore_tol::Real
    restore_tol_mode::Symbol
    restore_prox::Real
    pbm_adaptive::Bool
    keep_all::Bool
end
function AugLagOptions(;
    sub_options=nothing,
    relaxed=RelaxConstraintSelection(),
    penalty::Symbol=:quadratic,
    rho_init::Real=10.0,
    rho_max::Real=1e10,
    rho_incr::Real=10.0,
    rho_decr::Real=10.0,
    auto_rho_init::Bool=false,
    curvature::Symbol=:lbfgs,
    prox::Real=0.0,
    max_iter::Int=1000,
    ctol::Real=1e-6,
    eta0::Real=0.5,
    omega0::Real=1.0,
    weights=nothing,
    lsq_multipliers::Bool=true,
    scaling::Bool=true,
    memory::Int=50,
    elastic::Bool=false,
    elastic_mu::Real=1e3,
    warm_continue::Bool=false,
    filter::Bool=true,
    watchdog::Bool=false,
    watchdog_tau::Real=0.5,
    watchdog_tol::Real=0.0,
    anderson_ineq::Bool=false,
    anderson_depth::Int=2,
    multiplier_relax::Bool=false,
    spectral_rho::Bool=false,
    dual_secant::Bool=false,
    stall_retry::Bool=true,
    selective::Bool=true,
    safeguard::Bool=true,
    two_way::Bool=true,
    levenberg::Bool=true,
    h0::Symbol=:penalty,
    hard_rho::Real=1e3,
    sub_tol_schedule::Symbol=:eta,
    sub_tol_rel::Real=0.1,
    first_sub_tol=1e-6,
    first_sub_iters::Int=1,
    max_time::Real=Inf,
    on_iteration=nothing,
    analytical_gradient::Bool=false,
    stop_kkt::Bool=true,
    kkt_tol::Real=1e-3,
    second_order=nothing,
    inexact_cg::Bool=false,
    outer_schedule::Symbol=:bm,

    two_level::Bool=false,
    two_level_iters::Int=5,
    two_level_tau::Real=1.0,
    box_barrier=missing,
    box_barrier_scale::Real=1.0,
    restore::Bool=false,
    restore_stall::Int=0,
    restore_tol::Real=1e-2,
    restore_tol_mode::Symbol=:relative,
    restore_prox::Real=0.0,
    pbm_adaptive::Bool=false,
    keep_all::Bool=false,
)
    penalty in (:quadratic, :pbm, :mixed, :logdet, :mixed_logdet) || throw(
        ArgumentError("penalty must be :quadratic, :pbm, :mixed, :logdet or :mixed_logdet"),
    )
    outer_schedule in (:default, :bm) ||
        throw(ArgumentError("outer_schedule must be :default or :bm"))
    sub_tol_schedule in (:fixed, :eta2, :eta, :relative) ||
        throw(ArgumentError("sub_tol_schedule must be :fixed, :eta2, :eta or :relative"))
    0 < sub_tol_rel < 1 || throw(ArgumentError("sub_tol_rel must be in (0, 1)"))
    (first_sub_tol === nothing || first_sub_tol > 0) ||
        throw(ArgumentError("first_sub_tol must be positive or nothing"))
    first_sub_iters >= 1 || throw(ArgumentError("first_sub_iters must be >= 1"))
    max_time > 0 || throw(ArgumentError("max_time must be positive"))
    kkt_tol > 0 || throw(ArgumentError("kkt_tol must be positive"))
    (second_order === nothing || second_order isa SecondOrderSelection) ||
        throw(ArgumentError("second_order must be a SecondOrderSelection or nothing"))
    two_level_iters >= 1 || throw(ArgumentError("two_level_iters must be >= 1"))
    two_level_tau > 0 || throw(ArgumentError("two_level_tau must be positive"))
    box_barrier_scale > 0 || throw(ArgumentError("box_barrier_scale must be positive"))
    (box_barrier === true || box_barrier === false || box_barrier === missing) ||
        throw(ArgumentError("box_barrier must be true, false or missing (automatic)"))
    curvature in (:lbfgs, :sr1, :newton, :hybrid, :split, :fdgrad) || throw(
        ArgumentError(
            "curvature must be :lbfgs, :sr1, :newton, :hybrid, :split or :fdgrad"
        ),
    )
    prox >= 0 || throw(ArgumentError("prox must be non-negative"))
    rho_init > 0 || throw(ArgumentError("rho_init must be positive"))
    rho_incr > 1 || throw(ArgumentError("rho_incr must be > 1"))
    rho_decr > 1 || throw(ArgumentError("rho_decr must be > 1"))
    hard_rho > 0 || throw(ArgumentError("hard_rho must be positive"))
    0 < watchdog_tau < 1 || throw(ArgumentError("watchdog_tau must be in (0, 1)"))
    watchdog_tol >= 0 || throw(ArgumentError("watchdog_tol must be non-negative"))
    anderson_depth >= 1 || throw(ArgumentError("anderson_depth must be >= 1"))
    restore_stall >= 0 || throw(ArgumentError("restore_stall must be >= 0"))
    restore_tol_mode in (:relative, :absolute) ||
        throw(ArgumentError("restore_tol_mode must be :relative or :absolute"))
    restore_tol > 0 || throw(ArgumentError("restore_tol must be positive"))
    if restore_tol_mode === :relative
        restore_tol < 1 || throw(ArgumentError("relative restore_tol must be < 1"))
    end
    restore_prox >= 0 || throw(ArgumentError("restore_prox must be non-negative"))
    return AugLagOptions(
        sub_options,
        relaxed,
        penalty,
        rho_init,
        rho_max,
        rho_incr,
        rho_decr,
        auto_rho_init,
        curvature,
        prox,
        max_iter,
        ctol,
        eta0,
        omega0,
        weights,
        lsq_multipliers,
        scaling,
        memory,
        elastic,
        elastic_mu,
        warm_continue,
        filter,
        watchdog,
        watchdog_tau,
        watchdog_tol,
        anderson_ineq,
        anderson_depth,
        multiplier_relax,
        spectral_rho,
        dual_secant,
        stall_retry,
        selective,
        safeguard,
        two_way,
        levenberg,
        h0,
        hard_rho,
        sub_tol_schedule,
        sub_tol_rel,
        first_sub_tol,
        first_sub_iters,
        max_time,
        on_iteration,
        analytical_gradient,
        stop_kkt,
        kkt_tol,
        second_order,
        inexact_cg,
        outer_schedule,
        two_level,
        two_level_iters,
        two_level_tau,
        box_barrier,
        box_barrier_scale,
        restore,
        restore_stall,
        restore_tol,
        restore_tol_mode,
        restore_prox,
        pbm_adaptive,
        keep_all,
    )
end

"""
    AugLagResult

The result of an [`AugLagAlg`](@ref) optimization: the `GenericResult`-style
fields `minimizer` and `minimum`, the (currently `nothing`) sub-solver
`results`, and `trajectory` — the vector of outer-loop iterates
`(x, f(x), ‖v(x)‖∞)` when `AugLagOptions(keep_all = true)`, `nothing`
otherwise.
"""
struct AugLagResult{TM1,TM2,R,TR} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    results::R
    trajectory::TR
    status::Symbol
end
function AugLagResult(minimizer, minimum, results, trajectory)
    return AugLagResult(minimizer, minimum, results, trajectory, :unknown)
end

mutable struct AugLagWorkspace{
    M<:VecModel,X<:AbstractVector,O<:AugLagOptions,S<:Union{Nothing,AbstractOptimizer}
} <: Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
    state::Any
end

function Workspace(
    model::VecModel,
    optimizer::AugLagAlg,
    x0::AbstractVector,
    args...;
    options::AugLagOptions=AugLagOptions(),
    kwargs...,
)
    if optimizer.sub_alg === nothing
        sel = options.relaxed
        nsd = length(_relaxed_indices(sel.sd, model.sd_constraints, sel.labels))
        ntotal_sd = length(model.sd_constraints.fs)
        nsd == ntotal_sd || throw(
            ArgumentError(
                "AugLagAlg() cannot keep semidefinite constraints hard for the " *
                "built-in QuasiNewtonAlg; relax them (`RelaxConstraintSelection(sd = true)`) " *
                "or pass a sub_alg.",
            ),
        )
        nsets = length(_relaxed_indices(sel.sets, model.set_constraints, sel.labels))
        ntotal_sets = length(model.set_constraints.fs)
        nsets == ntotal_sets || throw(
            ArgumentError(
                "AugLagAlg() cannot keep function-in-set constraints hard for the " *
                "built-in QuasiNewtonAlg; relax them (`RelaxConstraintSelection(sets = true)`) " *
                "or pass a sub_alg.",
            ),
        )
        # kept (non-relaxed) equality/inequality constraints are allowed:
        # they are folded into the built-in subproblem as a smooth quadratic
        # penalty with weight hard_rho (see the hard_rho option).
    end
    return AugLagWorkspace(model, copy(x0), options, optimizer.sub_alg, nothing)
end

# Convenience: default to the model's initial point when no x0 is given.
function Workspace(
    model::VecModel,
    optimizer::AugLagAlg,
    args...;
    options::AugLagOptions=AugLagOptions(),
    kwargs...,
)
    isempty(args) &&
        return Workspace(model, optimizer, getinit(model); options=options, kwargs...)
    return Workspace(
        model, optimizer, args[1], Base.tail(args)...; options=options, kwargs...
    )
end

# ------------------------------------------------------------------
# Subproblem construction
# ------------------------------------------------------------------
# The sub-solver options with the outer loop's current subproblem tolerance
# applied (the adaptive sub_tol_schedule governs every sub-solver). The
# fallback keeps the options unchanged; solver-specific methods live in the
# package extensions (NonconvexAugLagLabIpoptExt / -MMAExt / -NLoptExt).
_set_sub_tol(alg, options, tol) = options

# The inner (subproblem) tolerance of one outer iteration: an explicit
# sub_options.tol override wins; otherwise the schedule - :fixed (1e-8),
# :eta2 (clamp(eta^2, 1e-10, 1e-8)), :eta (clamp(eta^2, 1e-6, 1e-2)) or
# :relative (clamp(sub_tol_rel * vprev, 1e-10, 1e-2), the Eckstein-Silva
# practical relative criterion: the subproblem only needs to be solved to
# an accuracy proportional to the current constraint violation):
# loose inner solves while the outer loop only needs rough steps, with a
# 1e-6 floor for the final accuracy.
function _sub_tol_value(schedule, tol_override, eta, vprev, sub_tol_rel)
    tol_override !== nothing && return Float64(tol_override)
    schedule === :fixed && return 1e-8
    schedule === :eta2 && return clamp(eta^2, 1e-10, 1e-8)
    schedule === :eta && return clamp(eta^2, 1e-6, 1e-2)
    schedule === :relative && return clamp(sub_tol_rel * max(vprev, 1e-12), 1e-10, 1e-2)
    return throw(ArgumentError("sub_tol_schedule must be :fixed, :eta2, :eta or :relative"))
end

function _subproblem(
    model::VecModel, sel::RelaxConstraintSelection, weights::Union{Nothing,AbstractVector}
)
    eq_rel = _relaxed_indices(sel.eq, model.eq_constraints, sel.labels)
    ineq_rel = _relaxed_indices(sel.ineq, model.ineq_constraints, sel.labels)
    sd_rel = _relaxed_indices(sel.sd, model.sd_constraints, sel.labels)
    sets_rel = _relaxed_indices(sel.sets, model.set_constraints, sel.labels)
    kept_eq = VectorOfFunctions(
        EqConstraint[c for (i, c) in enumerate(model.eq_constraints.fs) if !(i in eq_rel)]
    )
    kept_ineq = VectorOfFunctions(
        IneqConstraint[
            c for (i, c) in enumerate(model.ineq_constraints.fs) if !(i in ineq_rel)
        ],
    )
    kept_sd = VectorOfFunctions(
        SDConstraint[c for (i, c) in enumerate(model.sd_constraints.fs) if !(i in sd_rel)]
    )
    kept_sets = VectorOfFunctions(
        FunctionInSet[
            c for (i, c) in enumerate(model.set_constraints.fs) if !(i in sets_rel)
        ],
    )
    return eq_rel,
    ineq_rel, sd_rel, sets_rel,
    VecModel(
        model.objective,
        kept_eq,
        kept_ineq,
        kept_sd,
        kept_sets,
        VariableInSet[],
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end

# ------------------------------------------------------------------
# Violation measure of the relaxed constraints
# ------------------------------------------------------------------
_vecval(v::Number) = [v]
_vecval(v::AbstractVector) = vec(v)

# Extension hook (overridden by NonconvexAugLagLabSDPExt)
function _sd_violation(model, i, x)
    return throw(
        ArgumentError(
            "Semidefinite constraints are selected for relaxation but " *
            "NonconvexSemidefinite is not loaded; load it to enable the " *
            "projected augmented Lagrangian for semidefinite constraints.",
        ),
    )
end

# ------------------------------------------------------------------
# Augmented Lagrangian objective (quadratic / PBM)
# ------------------------------------------------------------------
# PBM barrier term Σᵢ c log(1 + uᵢ gᵢ / c) for uᵢ gᵢ / c > -1 (Ben-Tal &
# Zibulevsky's penalty-barrier-multiplier). The argument is clamped at the
# domain boundary so deeply feasible points stay flat.
function _pbm_barrier(g, u, c)
    s = 0.0
    ε = 1e-8
    for i in eachindex(g)
        t = clamp(u[i] * g[i] / c, -1 + ε, 1e15)
        s += c * log(1 + t)
    end
    return s
end

function _unit_weights(weights, indices)
    n = length(indices)
    weights === nothing && return ones(n)
    length(weights) == n || throw(
        ArgumentError(
            "weights must have one entry per relaxed constraint group ($n), got $(length(weights)).",
        ),
    )
    return weights
end

function _auglag_objective(
    model::VecModel,
    sel::RelaxConstraintSelection,
    weights,
    ρ::Real,
    y,
    λ,
    sd_terms::Function,
    penalty::Symbol,
    b_ineq=nothing,
)
    eq_rel, ineq_rel, _, _, _ = _subproblem(model, sel, nothing)
    objective0 = model.objective
    objective0 === nothing && throw(ArgumentError("The model has no objective."))
    w_eq = _unit_weights(weights, eq_rel)
    w_ineq = _unit_weights(weights, ineq_rel)
    function _objective(x)
        target = objective0(x)
        for (kk, i) in enumerate(eq_rel)
            c = _vecval(model.eq_constraints.fs[i](x))
            r = ρ * w_eq[kk]
            target += dot(y[kk], c) + r / 2 * dot(c, c)
        end
        for (kk, i) in enumerate(ineq_rel)
            g = _vecval(model.ineq_constraints.fs[i](x))
            r = ρ * w_ineq[kk]
            cbar = b_ineq === nothing ? r : b_ineq[kk]
            if penalty in (:pbm, :mixed, :mixed_logdet)
                # PBM barrier on the feasible side, classical quadratic
                # augmented Lagrangian on the infeasible side (the
                # PENNON-style globalization of the barrier domain)
                feas = g .<= 0
                target += _pbm_barrier(g[feas], λ[kk][feas], cbar)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
            end
        end
        target += sd_terms(x)
        return target
    end
    return _objective
end

# Multiplier buffers per constraint group (one vector of multipliers per
# relaxed constraint).
function _init_multipliers(model, sel)
    eq_rel, ineq_rel, _, _, _ = _subproblem(model, sel, nothing)
    y = [zeros(length(_vecval(model.eq_constraints.fs[i](model.init)))) for i in eq_rel]
    λ = [zeros(length(_vecval(model.ineq_constraints.fs[i](model.init)))) for i in ineq_rel]
    return y, λ
end

# ------------------------------------------------------------------
# Problem scaling (Curtis, Jiang & Robinson style)
# ------------------------------------------------------------------
# The augmented Lagrangian penalty multiplies squared constraint values, so
# constraint groups with very different gradient magnitudes get very different
# curvatures and the subproblems become ill-conditioned (this is what breaks
# the classical scheme on problems like AC-OPF). Scale each relaxed group by
# the largest gradient magnitude at the initial point (capped below at 1) and
# the objective by its initial magnitude, then run the classical algorithm on
# the balanced problem; the returned objective is unscaled.
function _group_scales(model::VecModel, sel, x0)
    eq_rel, ineq_rel, _, _, _ = _subproblem(model, sel, nothing)
    scales = Float64[]
    # Gradient-magnitude scale from one finite-difference probe per group:
    # ‖c(x₀ + δ) - c(x₀)‖ / ‖δ‖ estimates the group's gradient magnitude
    # without materializing any Jacobian (a full Jacobian of a large
    # vector-valued constraint would be an N×N dense matrix).
    δ = 1e-6 .* (1 .+ abs.(x0))
    δnorm = norm(δ)
    for (kind, rel) in (("eq", eq_rel), ("ineq", ineq_rel))
        fs = kind == "eq" ? model.eq_constraints.fs : model.ineq_constraints.fs
        for i in rel
            c = fs[i]
            c0 = _vecval(c(x0))
            c1 = _vecval(c(x0 .+ δ))
            d = norm(c1 .- c0) / δnorm
            push!(scales, max(1.0, d))
        end
    end
    return eq_rel, ineq_rel, scales
end

function _scaled_problem(model::VecModel, sel, x0)
    eq_rel, ineq_rel, scales = _group_scales(model, sel, x0)
    # Scaled constraint functions: c̃(x) = c(x) / d.
    eq_fs = [
        (x -> _vecval(model.eq_constraints.fs[eq_rel[i]](x)) ./ scales[i]) for
        i in eachindex(eq_rel)
    ]
    ineq_fs = [
        (
            x ->
                _vecval(model.ineq_constraints.fs[ineq_rel[i]](x)) ./
                scales[length(eq_rel) + i]
        ) for i in eachindex(ineq_rel)
    ]
    return eq_fs, ineq_fs, scales
end

# Scaled augmented Lagrangian objective: same structure as the classical one,
# evaluated on the pre-scaled constraint functions and scaled objective.
function _auglag_objective_scaled(
    model::VecModel,
    sel,
    weights::Tuple,
    ρ::Real,
    y,
    λ,
    sd_terms::Function,
    penalty::Symbol,
    objective0,
    eq_fs,
    ineq_fs,
    b_ineq=nothing,
)
    w_eq, w_ineq = weights
    function _objective(x)
        target = objective0(x)
        for (kk, c) in enumerate(eq_fs)
            cv = _vecval(c(x))
            r = ρ * w_eq[kk]
            target += dot(y[kk], cv) + r / 2 * dot(cv, cv)
        end
        for (kk, c) in enumerate(ineq_fs)
            g = _vecval(c(x))
            r = ρ * w_ineq[kk]
            cbar = b_ineq === nothing ? r : b_ineq[kk]
            if penalty in (:pbm, :mixed, :mixed_logdet)
                # PBM barrier on the feasible side, classical quadratic
                # augmented Lagrangian on the infeasible side.
                feas = g .<= 0
                target += _pbm_barrier(g[feas], λ[kk][feas], cbar)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
            end
        end
        target += sd_terms(x)
        return target
    end
    return _objective
end

# Exact Gauss-Newton penalty terms of the relaxed constraint groups at x
# (for curvature = :hybrid): rho*w*J'J per group, matrix-free, with the
# inequality rows restricted to the active set frozen at x. JuMP-converted
# blocks use the exact sparse evaluator Jacobian (two sparse matvecs per
# CG iteration, one Jacobian evaluation per rebound); the rest use the
# finite-difference matvec. The evaluator's row order is inequalities
# first, then equalities; the scaling folds into the jump weights
# (c = s*c~ so rho*w*J~'J~ = (rho*w/s^2) J'J).
function _hybrid_terms(
    x,
    model,
    eq_rel,
    ineq_rel,
    eq_fs,
    ineq_fs,
    w_eq,
    w_ineq,
    ρ,
    je,
    scales,
    so_eq,
    so_ineq,
    y,
    λ,
)
    neq = length(eq_rel)
    terms = ()
    # evaluator-length multiplier vector (row order: inequalities first,
    # then equalities) for the label-gated exact curvature terms
    μ = if je !== nothing && (any(so_ineq) || any(so_eq))
        zeros(je.nineq + je.neq)
    else
        nothing
    end
    off = 0
    for (k, i) in enumerate(ineq_rel)
        gv = _vecval(ineq_fs[k](x))
        act = gv .> 0
        if je !== nothing && _is_jump_block(model.ineq_constraints.fs[i], je)
            rawg = _vecval(model.ineq_constraints.fs[i](x))
            rows = (off + 1):(off + length(rawg))
            w = ρ * w_ineq[k] / scales[neq + k]^2
            terms = (terms..., JumpGaussNewtonTerm(je, rows, w, copy(x), act))
            if so_ineq[k]
                s = scales[neq + k]
                for (j, r) in enumerate(rows)
                    act[j] && (μ[r] = (λ[k][j] + ρ * w_ineq[k] * gv[j]) / s)
                end
            end
        else
            terms = (terms..., GaussNewtonTerm(ineq_fs[k], ρ * w_ineq[k], copy(x), act))
        end
        off += length(gv)
    end
    for (k, i) in enumerate(eq_rel)
        if je !== nothing && _is_jump_block(model.eq_constraints.fs[i], je)
            rawc = _vecval(model.eq_constraints.fs[i](x))
            rows = (off + 1):(off + length(rawc))
            w = ρ * w_eq[k] / scales[k]^2
            terms = (terms..., JumpGaussNewtonTerm(je, rows, w, copy(x)))
            if so_eq[k]
                s = scales[k]
                cv = _vecval(eq_fs[k](x))
                for (j, r) in enumerate(rows)
                    μ[r] = (y[k][j] + ρ * w_eq[k] * cv[j]) / s
                end
            end
        else
            terms = (terms..., GaussNewtonTerm(eq_fs[k], ρ * w_eq[k], copy(x)))
        end
        off += length(_vecval(eq_fs[k](x)))
    end
    μ === nothing || (terms = (terms..., JumpLagrangianTerm(je, μ, copy(x))))
    return terms
end

# The penalty part phi = rho/2 (||c||^2 + ||g^+||^2) of the scaled AL
# objective and its gradient (one reverse pass), for the split-secant
# curvature model: the penalty secant pair uses the exact gradient
# difference, and the remainder receives the corrected pair.
function _penalty_closure(eq_fs, ineq_fs, w_eq, w_ineq, ρ)
    return x -> begin
        v = 0.0
        for (k, c) in enumerate(eq_fs)
            cv = _vecval(c(x))
            v += (ρ * w_eq[k] / 2) * dot(cv, cv)
        end
        for (k, c) in enumerate(ineq_fs)
            gp = max.(_vecval(c(x)), 0.0)
            v += (ρ * w_ineq[k] / 2) * dot(gp, gp)
        end
        return v
    end
end
function _penalty_gradient(eq_fs, ineq_fs, w_eq, w_ineq, ρ)
    φ = _penalty_closure(eq_fs, ineq_fs, w_eq, w_ineq, ρ)
    return x -> NonconvexCore.value_gradient(φ, x)[2]
end

# Smooth quadratic penalty over the KEPT (non-relaxed) constraints, folded
# into the built-in quasi-Newton subproblem objective with weight hard_rho
# (the ALM-for-relaxed / penalty-for-kept split; the kept constraints are
# then removed from the submodel so the box-only solver accepts it). The
# feasibility of the kept constraints is traded against hard_rho (an exact
# penalty argument: large hard_rho drives their violation to zero).
function _hard_penalty_objective(obj, kept_eq, kept_ineq, hard_rho, n)
    return z -> begin
        v = obj(z)
        x = length(z) == n ? z : @view(z[1:n])
        for c in kept_eq.fs
            cv = _vecval(c(x))
            v += (hard_rho / 2) * dot(cv, cv)
        end
        for c in kept_ineq.fs
            gv = max.(_vecval(c(x)), 0.0)
            v += (hard_rho / 2) * dot(gv, gv)
        end
        return v
    end
end

# Analytical gradient path for the built-in quasi-Newton subproblems (#32):
# the AL objective's gradient is assembled as grad f + one J'v per
# constraint group (vector-Jacobian products through the existing
# reverse-mode machinery, no materialized Jacobian), and exposed to AD via
# a ChainRules rrule (the same pattern the SDP extension uses for the cone
# projection), so the prox / box-barrier / hard-penalty wrappers keep
# working. Quadratic penalty mode only (the PBM branch has its own
# closed-form derivative, future work).
struct AnalyticalALObjective{TF,TE,TI,TY,TL,TW,TS} <: Function
    scaled_obj::TF
    eq_fs::TE
    ineq_fs::TI
    y::TY
    λ::TL
    w_eq::TW
    w_ineq::TW
    ρ::Float64
    sd_terms::TS
end

function (a::AnalyticalALObjective)(x)
    v = a.scaled_obj(x)
    for (k, c) in enumerate(a.eq_fs)
        cv = _vecval(c(x))
        r = a.ρ * a.w_eq[k]
        v += dot(a.y[k], cv) + r / 2 * dot(cv, cv)
    end
    for (k, c) in enumerate(a.ineq_fs)
        gv = _vecval(c(x))
        gp = max.(gv, 0.0)
        r = a.ρ * a.w_ineq[k]
        v += dot(a.λ[k], gp) + r / 2 * dot(gp, gp)
    end
    return v + a.sd_terms(x)
end

# ∇L(x) = ∇f(x) + Σ Jᵢ(x)ᵀ (yᵢ + ρ wᵢ c̃ᵢ(x)) + Σ Jⱼ(x)ᵀ (λⱼ + ρ wⱼ gⱼ⁺)
# (on the violated rows only) + ∇(sd terms); each Jᵀv is one reverse pass.
function _analytical_al_gradient(a::AnalyticalALObjective, x)
    _, ∇f = NonconvexCore.value_gradient(a.scaled_obj, x)
    g = copy(∇f)
    for (k, c) in enumerate(a.eq_fs)
        cv = _vecval(c(x))
        v = a.y[k] .+ (a.ρ * a.w_eq[k]) .* cv
        gJ = NonconvexCore.value_gradient(z -> dot(_vecval(c(z)), v), x)[2]
        gJ === nothing || (g .+= gJ)
    end
    for (k, c) in enumerate(a.ineq_fs)
        gv = _vecval(c(x))
        act = gv .> 0
        if any(act)
            v = zero(gv)
            v[act] = a.λ[k][act] .+ (a.ρ * a.w_ineq[k]) .* gv[act]
            gJ = NonconvexCore.value_gradient(z -> dot(_vecval(c(z)), v), x)[2]
            gJ === nothing || (g .+= gJ)
        end
    end
    # the sd terms carry their own exact rrule (the cone-projection one)
    gsd = NonconvexCore.value_gradient(a.sd_terms, x)[2]
    gsd === nothing || (g .+= gsd)
    return g
end

function ChainRulesCore.rrule(a::AnalyticalALObjective, x)
    val = a(x)
    function analytical_pullback(Δ)
        g = _analytical_al_gradient(a, x)
        return (ChainRulesCore.NoTangent(), Δ .* g)
    end
    return val, analytical_pullback
end

# Violation measure of the relaxed constraints on the scaled functions.
# The relaxed indices are passed in (they are computed once per solve;
# recomputing the subproblem decomposition per call is a hot-path cost).
function _scaled_violation(eq_fs, ineq_fs, sd_rel, sets_rel, model, x, set_scales)
    v = Float64[]
    for c in eq_fs
        append!(v, abs.(_vecval(c(x))))
    end
    for c in ineq_fs
        append!(v, max.(_vecval(c(x)), 0.0))
    end
    for i in sd_rel
        append!(v, _sd_violation(model, i, x))
    end
    for (k, i) in enumerate(sets_rel)
        append!(v, _set_violation(model, i, x, set_scales[k]))
    end
    return v
end

# Raw scaled constraint values at x (signed equalities, inequalities as-is)
# — the ingredients of the objective recovery and the violation norm, so
# the hot path evaluates each relaxed constraint exactly once.
function _scaled_constraint_values(eq_fs, ineq_fs, x)
    eqvals = [_vecval(c(x)) for c in eq_fs]
    ineqvals = [_vecval(c(x)) for c in ineq_fs]
    return eqvals, ineqvals
end

# Violation vector from precomputed scaled values (without the sd part).
function _scaled_violation_from(eqvals, ineqvals)
    v = Float64[]
    for cv in eqvals
        append!(v, abs.(cv))
    end
    for gv in ineqvals
        append!(v, max.(gv, 0.0))
    end
    return v
end

# Recover the (unscaled) objective from the AL value and the raw scaled
# constraint values at the same point (classical quadratic mode):
#   f = fscale · (L_ρ − Σ (yᵀc̃ + ρw/2‖c̃‖²) − Σ (λᵀg̃⁺ + ρw/2‖g̃⁺‖²)).
# Saves the extra objective and constraint evaluations of re-evaluating
# obj(x) after every accepted step (Percival's objective recovery).
function _recover_objective(mt, eqvals, ineqvals, y, λ, w_eq, w_ineq, ρ, fscale)
    f = mt
    for (k, cv) in enumerate(eqvals)
        r = ρ * w_eq[k]
        f -= dot(y[k], cv) + r / 2 * dot(cv, cv)
    end
    for (k, gv) in enumerate(ineqvals)
        gp = max.(gv, 0.0)
        r = ρ * w_ineq[k]
        f -= dot(λ[k], gp) + r / 2 * dot(gp, gp)
    end
    return fscale == 1.0 ? f : f * fscale
end

# ------------------------------------------------------------------
# Outer loop
# ------------------------------------------------------------------
# Anderson mixing on the equality multiplier fixed-point update: given the
# last `m` iterates yⱼ with residuals Rⱼ = G(yⱼ) − yⱼ (G the multiplier
# update map), pick the simplex weights θ (Σθⱼ = 1) minimizing the norm of
# the mixed residual ΣθⱼRⱼ and take y ← Σθⱼ(yⱼ + Rⱼ). Cheap (m × m), and
# typically halves the outer iteration count.
function _anderson_mix(history, y_new)
    m = length(history)
    m == 0 && return y_new
    Rs = [h[2] for h in history]
    Ys = [h[1] for h in history]
    m == 1 && return y_new
    R = Rs[end]
    # With θₘ = 1 − Σ_{j<m} θⱼ the mixed residual is R + Σ_{j<m} θⱼ (Rⱼ − R),
    # so the normal equations of min ‖ΣθⱼRⱼ‖² over the simplex are
    #   G θ = b,  Gᵢⱼ = ⟨Rᵢ − R, Rⱼ − R⟩,  bᵢ = −⟨Rᵢ − R, R⟩.
    A = [Rs[j] .- R for j in 1:(m - 1)]
    G = zeros(m - 1, m - 1)
    rhs = zeros(m - 1)
    for i in 1:(m - 1), j in 1:(m - 1)
        G[i, j] = dot(A[i], A[j])
    end
    for i in 1:(m - 1)
        rhs[i] = -dot(A[i], R)
    end
    θf = try
        clamp.(G \ rhs, -2.0, 2.0)
    catch
        # singular normal equations (e.g. identical residuals): keep the
        # unaccelerated update
        zeros(m - 1)
    end
    θ = ones(m) ./ m
    θ[1:(m - 1)] .= θf
    θ[m] = 1 - sum(θf)
    out = zeros(length(y_new))
    for j in 1:m
        out .+= θ[j] .* (Ys[j] .+ Rs[j])
    end
    return out
end

# Fallback for a rejected outer iterate: the previous trajectory point, or
# the run's initial point when the very first iteration fails (the trajectory
# is still empty then and `trajectory[end]` would throw a BoundsError).
# Raw (unscaled) violation norm of the relaxed constraints — the physical
# feasibility metric used to accept restoration points and to choose
# between the final main-phase iterate and the best restoration point
# (the restoration merit minimizes this, while the main phase's internal
# metric is the scaled violation).
function _raw_violation(model, eq_rel, ineq_rel, sets_rel, x)
    v = 0.0
    for i in eq_rel
        v = max(v, norm(_vecval(model.eq_constraints.fs[i](x)), Inf))
    end
    for i in ineq_rel
        v = max(v, norm(max.(_vecval(model.ineq_constraints.fs[i](x)), 0.0), Inf))
    end
    for i in sets_rel
        v = max(v, norm(_set_violation(model, i, x), Inf))
    end
    return v
end

# Compatibility wrapper for the selection-based call (the hot paths pass
# the precomputed indices).
function _raw_violation(model, relaxed, x)
    eq_rel, ineq_rel, _, sets_rel, _ = _subproblem(model, relaxed, nothing)
    return _raw_violation(model, eq_rel, ineq_rel, sets_rel, x)
end

function _failed_iterate_fallback(trajectory, x_init)
    return isempty(trajectory) ? copy(x_init) : copy(trajectory[end][1])
end

# Restoration phase (Ipopt-style): solve a feasibility-only problem
# (squared violations of the relaxed constraints over the box) to recover a
# usable iterate after a blow-up, instead of just reverting.
function _restore_feasible(eq_fs, ineq_fs, model, relaxed, submodel0, x, prox=0.0)
    x̄ = copy(x)  # snapshot: the anchor must not drift with the solve
    eq_rel, ineq_rel, sd_rel, sets_rel, _ = _subproblem(model, relaxed, nothing)
    # anchor widths: relative moves are measured against each variable's
    # box width (a raw Euclidean anchor would be dominated by the widest
    # block, e.g. displacements in [-1e3, 1e3] vs densities in [0, 1])
    wscale = map(
        (lo, hi) -> isfinite(lo) && isfinite(hi) && hi > lo ? hi - lo : 1.0,
        submodel0.box_min,
        submodel0.box_max,
    )
    objr =
        z -> begin
            v = 0.0
            # RAW constraint values: the restoration merit is the physical
            # feasibility metric, not the solver's internal scaled one (the
            # CJR scales weight the groups by their Jacobian norms, which on
            # problems like SAND makes a small-scale group like the volume
            # constraint dominate the merit while the equilibrium group —
            # the physically meaningful one — is ignored). Variable-level
            # conditioning is handled by the formulation's diagonal scaling
            # below instead.
            for i in eq_rel
                v += sum(abs2, _vecval(model.eq_constraints.fs[eq_rel[i]](z)))
            end
            for i in ineq_rel
                v += sum(abs2, max.(_vecval(model.ineq_constraints.fs[ineq_rel[i]](z)), 0.0))
            end
            for i in sd_rel
                v += sum(abs2, _sd_violation(model, i, z))
            end
            for i in sets_rel
                v += sum(abs2, _set_violation(model, i, z))
            end
            # optional Rockafellar anchor: stay close to the pre-restoration
            # iterate (the violation merit alone can wander along its zero
            # manifold). Per-variable scaled pseudo-Huber (log-cosh) penalty:
            # f_i = w_i * log(cosh(Delta_i / w_i)) has gradient
            # tau * tanh(Delta_i / w_i) — magnitude < tau for EVERY variable,
            # saturating instead of growing linearly, with the transition
            # width scaled per variable by its box width (a raw Euclidean
            # anchor would be dominated by the widest block, e.g.
            # displacements in [-1e3, 1e3] vs densities in [0, 1]).
            if prox > 0
                Δ = clamp.((z .- x̄) ./ wscale, -50.0, 50.0)
                v += prox * sum(wscale .* log.(cosh.(Δ)))
            end
            return v
        end
    # Diagonal inverse premultiplier IN THE FORMULATION (not the solver):
    # d_i = sum_j (dF_j/dz_i)^2 over the stacked RAW merit F = [c; g^+] —
    # the diagonal of the least-squares Hessian J'J. The restoration
    # submodel is defined in the scaled variables z = Ds .* zhat with
    # Ds_i = 1/sqrt(max(d_i, eps)), so every column of J * diag(Ds) has
    # unit norm: the raw merit would otherwise be dominated by the most
    # sensitive blocks (e.g. the stiffness-column dominated displacement
    # block in SAND, whose J'J conditioning stalls an unscaled
    # first-order solve).
    nz = length(x)
    hfd = sqrt(eps(Float64)) .* (1 .+ abs.(x))
    dcol = zeros(nz)
    for i in 1:nz
        zp = copy(x)
        zp[i] += hfd[i]
        zm = copy(x)
        zm[i] -= hfd[i]
        for k in eq_rel
            cp = _vecval(model.eq_constraints.fs[k](zp))
            cm = _vecval(model.eq_constraints.fs[k](zm))
            δ = (cp .- cm) ./ (2 * hfd[i])
            dcol[i] += sum(abs2, δ)
        end
        for k in ineq_rel
            gp = max.(_vecval(model.ineq_constraints.fs[k](zp)), 0.0)
            gm = max.(_vecval(model.ineq_constraints.fs[k](zm)), 0.0)
            δ = (gp .- gm) ./ (2 * hfd[i])
            dcol[i] += sum(abs2, δ)
        end
    end
    Ds = 1 ./ sqrt.(max.(dcol, 1e-12))
    rm = VecModel(
        Objective(zhat -> objr(Ds .* zhat)),
        VectorOfFunctions(NonconvexCore.EqConstraint[]),
        VectorOfFunctions(NonconvexCore.IneqConstraint[]),
        VectorOfFunctions(NonconvexCore.SDConstraint[]),
        submodel0.box_min ./ Ds,
        submodel0.box_max ./ Ds,
        submodel0.init ./ Ds,
        submodel0.integer,
    )
    r = optimize(
        rm,
        QuasiNewtonAlg(
            LBFGSApprox(50; H0=IdentityPreconditioner());
            max_iter=2000,
            tol=1e-10,
            globalization=:trustregion,
        ),
        x ./ Ds,
    )
    return Ds .* copy(r.minimizer)
end

# Two-way penalty decrease (Curtis-Gould-Jiang-Robinson): once feasibility
# is attained with an escalated penalty, back rho down toward rho_init so
# the subproblems stay well conditioned. The no-op case (feasibility
# reached without an escalated penalty, or the two-way update disabled)
# returns rho unchanged.
function _two_way_step(ρ, options, two_way, vnorm, η)
    if two_way && vnorm <= η && ρ > options.rho_init
        return max(ρ / options.rho_decr, options.rho_init)
    end
    return ρ
end

# Broyden secant correction on the equality multiplier map (Birgin-Martinez
# "Improving ultimate convergence" device): C is the inverse secant model of
# the Jacobian of the residual map c(x(y)). It starts as -I (so the first
# corrected step reproduces the plain ascent y + r) and is refined with the
# good-Broyden inverse update (Sherman-Morrison) from the secant pairs
# (dy, dr) of the multiplier map. The corrected step y - C*r replaces the
# plain ascent, safeguarded: the model step must not exceed twice the plain
# step's norm, and degenerate denominators leave C unchanged.
function _dual_secant_init!(bdual::Ref{Any}, m::Int)
    bdual[] = -Matrix{Float64}(I, m, m)
    return bdual
end

function _dual_secant_update!(C, Δy, Δc)
    dydy = dot(Δy, Δy)
    dydy > 0 || return C
    Cdc = C * Δc
    den = dot(Δy, Cdc)
    abs(den) > 1e-12 * norm(Δy) * norm(Cdc) || return C
    # good Broyden inverse update (Sherman-Morrison):
    #   C += (Δy - C Δc)(Δy'C)/(Δy'CΔc)
    u = (Δy .- Cdc) ./ den
    v = C' * Δy
    C .+= u .* v'
    return C
end

# Feasibility-driven penalty update on a bad step: escalate the most
# violated constraint groups (selective weights, when enabled) or the
# global penalty when no single group stands out.
function _escalate_penalty!(
    ρ, η, ω, w_eq, w_ineq, w_max, options, eq_rel, ineq_rel, eq_fs, ineq_fs, x
)
    bumped = false
    if options.selective
        for k in eachindex(eq_rel)
            vk = norm(_vecval(eq_fs[k](x)), Inf)
            if vk > η && w_eq[k] < w_max
                w_eq[k] = min(w_eq[k] * options.rho_incr, w_max)
                bumped = true
            end
        end
        for k in eachindex(ineq_rel)
            vk = norm(max.(_vecval(ineq_fs[k](x)), 0.0), Inf)
            if vk > η && w_ineq[k] < w_max
                w_ineq[k] = min(w_ineq[k] * options.rho_incr, w_max)
                bumped = true
            end
        end
    end
    if !bumped
        ρ = min(options.rho_max, ρ * options.rho_incr)
        if options.outer_schedule === :bm
            η = max(options.eta0 / ρ, options.ctol)
            ω = max(1 / ρ, sqrt(eps(Float64)))
        else
            η = max(options.eta0 / ρ^0.1, options.ctol)
            ω = max(1 / ρ^0.1, sqrt(eps(Float64)))
        end
    end
    return ρ, η, ω
end

function optimize!(workspace::AugLagWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack relaxed, penalty, max_iter, ctol, eta0, omega0 = options
    @unpack lsq_multipliers, scaling, keep_all, sub_options, memory = options
    @unpack elastic, elastic_mu, warm_continue = options
    @unpack filter, stall_retry, selective, safeguard, two_way = options
    anderson_ineq = options.anderson_ineq
    anderson_depth = options.anderson_depth
    multiplier_relax = options.multiplier_relax
    spectral_rho = options.spectral_rho
    dual_secant = options.dual_secant
    watchdog = options.watchdog
    watchdog_tau = Float64(options.watchdog_tau)
    watchdog_tol = Float64(options.watchdog_tol)
    levenberg = options.levenberg
    @unpack h0, sub_tol_schedule = options
    hard_rho = Float64(options.hard_rho)
    sub_tol_rel = Float64(options.sub_tol_rel)
    first_sub_tol =
        options.first_sub_tol === nothing ? nothing : Float64(options.first_sub_tol)
    first_sub_iters = options.first_sub_iters
    max_time = Float64(options.max_time)
    on_iteration = options.on_iteration
    analytical_gradient = options.analytical_gradient
    stop_kkt = options.stop_kkt
    kkt_tol = Float64(options.kkt_tol)
    inexact_cg = options.inexact_cg
    outer_schedule = options.outer_schedule
    @unpack two_level, two_level_iters, two_level_tau = options
    @unpack box_barrier_scale = options
    restore = options.restore
    restore_stall = options.restore_stall
    restore_tol = Float64(options.restore_tol)
    restore_tol_mode = options.restore_tol_mode
    restore_prox = Float64(options.restore_prox)
    pbm_adaptive = options.pbm_adaptive
    prox = Float64(options.prox)
    # automatic box-barrier selection by subproblem order: on with the
    # exact-Hessian (`newton`) subproblems, off with the first-order
    # quasi-Newton subproblems (measured 5.3x faster / harmful resp.)
    box_barrier = if options.box_barrier === missing
        options.curvature === :newton
    else
        options.box_barrier
    end
    obj = model.objective
    obj === nothing && throw(ArgumentError("The model has no objective."))
    ρ = Float64(options.rho_init)
    x = copy(x0)
    # Warm continuation restarts from the previous final iterate so the
    # constraint scales and multipliers are computed at the carried-over
    # design rather than the original x0.
    if warm_continue && workspace.state !== nothing && workspace.state.n == length(x)
        x = copy(workspace.state.x)
    end
    # The starting point of this run, for the first-iteration failure fallback.
    x_init = copy(x)
    eq_rel, ineq_rel, sd_rel, _, submodel0 = _subproblem(model, relaxed, options.weights)
    # The relaxed function-in-set constraints are handled as dedicated
    # groups: one bundle evaluation per call feeds all their components.
    sets_rel = _relaxed_indices(relaxed.sets, model.set_constraints, relaxed.labels)
    # Objective scale: minimize f/s_f so the AL objective terms are O(1).
    # (The PBM barrier parameters are calibrated to the unscaled objective,
    # so only the classical quadratic mode scales f.)
    fscale = 1.0
    if scaling && penalty === :quadratic
        f0 = obj(x)
        fscale = max(1.0, abs(f0))
    end
    scaled_obj = fscale == 1.0 ? obj : (x -> obj(x) / fscale)
    y, λ = _init_multipliers(model, relaxed)
    if scaling
        eq_fs, ineq_fs, scales = _scaled_problem(model, relaxed, x)
    else
        eq_fs = [model.eq_constraints.fs[i] for i in eq_rel]
        ineq_fs = [model.ineq_constraints.fs[i] for i in ineq_rel]
        scales = ones(length(eq_rel) + length(ineq_rel))
    end
    if options.auto_rho_init
        # calibrate ρ so the penalty gradient balances the objective gradient
        # at x0 (on the scaled constraints): avoids the escalation waste of a
        # too-small rho_init (measured: ~130 outer iterations on AC-OPF).
        _, ∇f = NonconvexCore.value_gradient(obj, x)
        viol0 = 0.0
        for c in eq_fs
            viol0 = max(viol0, norm(_vecval(c(x)), Inf))
        end
        for c in ineq_fs
            viol0 = max(viol0, norm(max.(_vecval(c(x)), 0.0), Inf))
        end
        ρ = clamp(norm(∇f) / max(viol0, 1e-8), 1.0, 1e6)
    end
    # Internal per-group penalty weights (selective bumps modify these; the
    # user-provided weights are the initial values).
    w_eq = _unit_weights(options.weights, 1:length(eq_fs))
    w_ineq = _unit_weights(options.weights, 1:length(ineq_fs))
    # Per-group PBM barrier coefficients (the adaptive barrier tracks the
    # complementarity of the implied duals instead of the global penalty).
    b_ineq = Float64[ρ * w_ineq[k] for k in eachindex(w_ineq)]
    n = length(x)
    m_eq = sum(c -> length(_vecval(c(x))), eq_fs; init=0)
    use_elastic = elastic && m_eq > 0
    # Warm continuation: reuse the previous run's multipliers (rescaled to
    # the new constraint scales), per-group penalty weights and quasi-Newton
    # curvature when the structure matches (e.g. across SIMP continuation
    # stages where the model or the AL hyper-parameters change between
    # solves). ρ, η, ω, the filter and the Anderson history are re-initialized
    # from the new options.
    warm = warm_continue && workspace.state !== nothing
    if warm
        st = workspace.state
        (
            st.n == n &&
            st.m_eq == m_eq &&
            st.neq == length(eq_fs) &&
            st.nineq == length(ineq_fs) &&
            st.elastic == use_elastic &&
            st.eq_rel == eq_rel &&
            st.ineq_rel == ineq_rel &&
            st.sd_rel == sd_rel
        ) || (warm = false)
    end
    if warm
        st = workspace.state
        for k in eachindex(eq_rel)
            y[k] = st.y[k] .* (scales[k] / st.scales[k])
        end
        for k in eachindex(ineq_rel)
            λ[k] = st.λ[k] .* (scales[length(eq_rel) + k] / st.scales[length(eq_rel) + k])
        end
        w_eq .= st.w_eq
        w_ineq .= st.w_ineq
    else
        if penalty in (:pbm, :mixed, :mixed_logdet)
            # the PBM barrier term (u/c) log(1 + c g) is inactive for u = 0;
            # initialize the barrier multipliers positive
            for k in eachindex(λ)
                λ[k] .= 1.0
            end
        end
        if lsq_multipliers && !isempty(eq_rel)
            y = _lsq_multipliers_scaled(scaled_obj, eq_fs, x)
        end
    end
    η = max(eta0, ctol)
    ω = max(omega0, 1 / ρ^0.1)
    w_max = Float64(options.rho_max) / ρ
    Y = _sd_init(model, sd_rel, ρ)
    # Multiplier buffers of the relaxed function-in-set constraints
    # (y_eq, λ_ineq, Y_psd per constraint).
    Msets = [_set_init_multipliers(model, i, ρ) for i in sets_rel]
    # The set-constraint component scales (gradient-magnitude probes, like
    # the classical group scales): the set penalties, violations and
    # multipliers live in these scaled units.
    set_scales = [_set_scales(model, i, x) for i in sets_rel]
    trajectory = Tuple{typeof(x),Real,Real}[]
    fx = obj(x)
    vprev = Inf
    vnorm = Inf
    and_history = Tuple{Vector{Float64},Vector{Float64}}[]
    # the AL objective ref for the finite-difference-gradient curvature
    # (updated each outer iteration once the subproblem is built)
    al_obj_ref = Ref{Any}(nothing)
    and_ineq_history = Tuple{Vector{Float64},Vector{Float64}}[]
    # spectral-penalty history: the flat constraint values and multipliers
    # of the previous good step (the Barzilai-Borwein estimate uses the
    # last two iterates of the dual map).
    cflat_prev = nothing
    yflat_prev = nothing
    # Broyden dual-secant state (equality multiplier map)
    bdual = Ref{Any}(nothing)
    ysec_prev = nothing
    csec_prev = nothing
    # infeasibility verdict (Percival's dual certificate)
    infeasible_flag = false
    # Ipopt-style restoration accounting: window of recent violation values
    # (for the stall trigger) and the number of restorations already
    # performed this solve.
    vhist = Float64[]
    n_restores = 0
    # Best restoration point: the main phase is not feasibility-preserving,
    # so a restoration point can be strictly more feasible than the final
    # main-phase iterate — track it and report it if so (opt-in behavior).
    x_rest = nothing
    fx_rest = 0.0
    vraw_rest = Inf
    # Fletcher-Leyffer filter for the outer globalization: a step is
    # accepted unless a previous iterate dominates it in both the AL merit
    # and the violation.
    filter_f = Float64[]
    filter_v = Float64[]
    # Nonmonotone watchdog reference (Birgin–Martínez style): a recursive
    # mean of the accepted AL merits; a step passing the filter must also
    # stay below this reference when `watchdog` is on.
    watchdog_fbar = Inf
    # Built-in quasi-Newton subsolver state: created once per solve with
    # `reset = false` so the L-BFGS curvature of the previous subproblem
    # warm-starts the next one; reset whenever the penalty changes scale.
    qn_base_maxiter = 800
    qn_tol_override = nothing
    qn_globalization = :trustregion
    qn_reset = false
    qn_nonmonotone = false
    qn_soc = false
    qn_approx = nothing
    # The built-in quasi-Newton subproblem options (parsed unconditionally
    # so the loop sites are well-typed for JET; used only when
    # sub_alg === nothing — external sub-solver options are not named
    # tuples and keep the defaults).
    qn_opts_is_nt = sub_options === nothing || sub_options isa NamedTuple
    qn_opts = if sub_options === nothing
        (; max_iter=800, tol=1e-8, globalization=:trustregion)
    else
        sub_options
    end
    qn_base_maxiter = qn_opts_is_nt ? get(qn_opts, :max_iter, 800) : 800
    qn_tol_override = qn_opts_is_nt ? get(qn_opts, :tol, nothing) : nothing
    qn_globalization =
        qn_opts_is_nt ? get(qn_opts, :globalization, :trustregion) : :trustregion
    qn_reset = qn_opts_is_nt ? get(qn_opts, :reset, false) : false
    qn_nonmonotone = qn_opts_is_nt ? get(qn_opts, :nonmonotone, false) : false
    qn_soc = qn_opts_is_nt ? get(qn_opts, :soc, false) : false
    qn_projected_stop = qn_opts_is_nt ? get(qn_opts, :projected_stop, true) : true
    qn_cg_maxiter = qn_opts_is_nt ? get(qn_opts, :cg_maxiter, nothing) : nothing
    qn_precond = qn_opts_is_nt ? get(qn_opts, :precond, false) : false
    if sub_alg === nothing
        h0_op = if h0 === :penalty
            DiagonalPreconditioner(fill(1 / (1 + ρ), n + (use_elastic ? m_eq : 0)))
        elseif h0 === :identity
            IdentityPreconditioner()
        else
            throw(ArgumentError("h0 must be :penalty or :identity"))
        end
        # the JuMP evaluator behind a JuMP-converted objective, for the
        # exact-Jacobian hybrid terms
        je_hybrid =
            options.curvature === :hybrid ? _jump_evaluator(model.objective) : nothing
        # per-group second-order flags for the hybrid terms (label-gated)
        so_eq, so_ineq = _second_order_flags(options.second_order, model, eq_rel, ineq_rel)
        qn_approx = if warm && workspace.state.approx !== nothing
            st_approx = workspace.state.approx
            if st_approx isa CompositeHessianApprox
                # keep the warm-started quasi-Newton history but rebuild
                # the exact terms with this run's penalty state
                CompositeHessianApprox(
                    st_approx.qn,
                    x -> _hybrid_terms(
                        x,
                        model,
                        eq_rel,
                        ineq_rel,
                        eq_fs,
                        ineq_fs,
                        w_eq,
                        w_ineq,
                        ρ,
                        je_hybrid,
                        scales,
                        so_eq,
                        so_ineq,
                        y,
                        λ,
                    ),
                )
            elseif st_approx isa SplitSecantApprox
                SplitSecantApprox(
                    st_approx.qn_rem,
                    st_approx.qn_pen,
                    _penalty_gradient(eq_fs, ineq_fs, w_eq, w_ineq, ρ),
                )
            else
                st_approx
            end
        elseif options.curvature === :sr1
            LimitedMemorySR1Approx(memory; H0=h0_op)
        elseif options.curvature === :hybrid
            penalty === :quadratic || throw(
                ArgumentError("curvature = :hybrid requires penalty = :quadratic")
            )
            use_elastic && throw(
                ArgumentError("curvature = :hybrid does not support the elastic mode"),
            )
            # the exact Gauss-Newton terms already carry the penalty scale,
            # so the remainder model starts from the identity
            base_qn = LBFGSApprox(memory; H0=IdentityPreconditioner())
            CompositeHessianApprox(
                base_qn,
                x -> _hybrid_terms(
                    x,
                    model,
                    eq_rel,
                    ineq_rel,
                    eq_fs,
                    ineq_fs,
                    w_eq,
                    w_ineq,
                    ρ,
                    je_hybrid,
                    scales,
                    so_eq,
                    so_ineq,
                    y,
                    λ,
                ),
            )
        elseif options.curvature === :split
            penalty === :quadratic ||
                throw(ArgumentError("curvature = :split requires penalty = :quadratic"))
            use_elastic && throw(
                ArgumentError("curvature = :split does not support the elastic mode"),
            )
            # split-secant: the penalty part phi and the remainder psi get
            # separate L-BFGS histories (structured secant from gradient
            # differences alone - no exact Hessians), and the penalty
            # model's inverse is the CG preconditioner
            base_rem = LBFGSApprox(memory; H0=IdentityPreconditioner())
            base_pen = LBFGSApprox(memory; H0=DiagonalPreconditioner(fill(max(ρ, 1.0), n)))
            SplitSecantApprox(
                base_rem, base_pen, _penalty_gradient(eq_fs, ineq_fs, w_eq, w_ineq, ρ)
            )
        elseif options.curvature === :fdgrad
            penalty === :quadratic || throw(
                ArgumentError("curvature = :fdgrad requires penalty = :quadratic")
            )
            use_elastic && throw(
                ArgumentError("curvature = :fdgrad does not support the elastic mode"),
            )
            # inexact Newton from one-sided finite differences of the
            # Zygote-computed AL gradient (one extra gradient evaluation
            # per Hessian-vector product); pair with a small cg_maxiter
            FDGradHessianApprox(z -> NonconvexCore.value_gradient(al_obj_ref[], z)[2])
        else
            LBFGSApprox(memory; H0=h0_op)
        end
    end
    # Exact-Hessian subproblem solves for JuMP-converted models (opt-in):
    # the AL objective's Hessian comes from the MathOptInterface evaluator
    # for JuMP-converted models and from AD otherwise.
    ρref = Ref(ρ)
    jump_newton_active = false
    if options.curvature === :newton
        penalty === :quadratic ||
            throw(ArgumentError("newton requires penalty = :quadratic"))
        isempty(sd_rel) ||
            throw(ArgumentError("newton does not support semidefinite constraints"))
        use_elastic && throw(ArgumentError("newton does not support the elastic mode"))
        qn_globalization = :trustregion
        je = _jump_evaluator(model.objective)
        qn_approx = if je !== nothing
            # JuMP-converted: exact evaluator Hessians with the
            # stale-Cholesky-preconditioned Steihaug CG.
            jump_newton_active = true
            JumpNewtonTRApprox(
                x -> begin
                    H = _jump_al_hessian(
                        je,
                        x,
                        eq_fs,
                        ineq_fs,
                        scales,
                        y,
                        λ,
                        w_eq,
                        w_ineq,
                        ρref,
                        fscale,
                        box_barrier ? submodel0.box_min : nothing,
                        box_barrier ? submodel0.box_max : nothing,
                        μref,
                        δbar,
                    )
                    # the prox term adds τI to the subproblem Hessian
                    prox > 0 && (H[diagind(H)] .+= prox)
                    H
                end,
            )
        else
            # other models: exact Hessians through AD (value_hessian)
            NewtonTRApprox()
        end
    end
    if two_level
        sub_alg === nothing ||
            throw(ArgumentError("two_level requires the built-in sub-solver (AugLagAlg())"))
        isempty(sd_rel) && throw(
            ArgumentError(
                "two_level requires relaxed semidefinite constraints " *
                "(RelaxConstraintSelection(sd = true))",
            ),
        )
    end
    if box_barrier && use_elastic
        throw(ArgumentError("box_barrier does not support the elastic mode"))
    end
    # Exact second-order plain subproblems for JuMP-converted models with
    # the Ipopt sub-solver: the subproblem Lagrangian Hessian is the AL
    # objective's Hessian assembled from the MathOptInterface evaluator
    # (nested AD through the JuMP functions is not supported). Requires the
    # quadratic penalty, no elastic mode, no semidefinite block, every
    # constraint block to belong to one evaluator, and none kept hard in the
    # subproblem.
    je_plain = nothing
    if penalty === :quadratic &&
        !use_elastic &&
        isempty(sd_rel) &&
        sub_alg !== nothing &&
        nameof(typeof(sub_alg)) === :IpoptAlg
        je = _jump_evaluator(model.objective)
        if je !== nothing &&
            all(i -> i in eq_rel, 1:length(model.eq_constraints.fs)) &&
            all(i -> i in ineq_rel, 1:length(model.ineq_constraints.fs)) &&
            all(_is_jump_block(model.eq_constraints.fs[i], je) for i in eq_rel) &&
            all(_is_jump_block(model.ineq_constraints.fs[i], je) for i in ineq_rel)
            je_plain = je
        end
    end
    δbar = 1e-4
    μref = Ref(0.0)
    if box_barrier
        # bound-push (Ipopt's bound_push): start strictly inside the finite
        # bounds by the barrier thickness, so the first subproblem sees the
        # log branch of the barrier rather than the quadratic extension
        for i in 1:n
            isfinite(submodel0.box_min[i]) &&
                (x[i] = max(x[i], submodel0.box_min[i] + δbar))
            isfinite(submodel0.box_max[i]) &&
                (x[i] = min(x[i], submodel0.box_max[i] - δbar))
        end
        μref[] = Float64(box_barrier_scale)
    end
    t0 = time()
    for iter in 1:max_iter
        # Wall-clock budget: no new subproblem is started once the budget
        # is exhausted (the elapsed check is per outer iteration).
        time() - t0 > max_time && break
        # Subproblem tolerance: fixed, adaptive (tied to the outer
        # feasibility target η), relative to the current violation, or an
        # explicit sub_options.tol override (previously silently ignored for
        # the built-in QN).
        sub_tol = _sub_tol_value(sub_tol_schedule, qn_tol_override, η, vprev, sub_tol_rel)
        # Optional first-subproblems tightening: the first inner solves
        # select the local basin the whole solve follows (measured on the
        # TopOpt local-stress problem: a loose first solve lands in a
        # heavier design's basin), so they can be solved more accurately
        # than the schedule without paying the tight tolerance on later
        # subproblems.
        if iter <= first_sub_iters && first_sub_tol !== nothing
            sub_tol = min(sub_tol, first_sub_tol)
        end
        # the barrier parameter relaxes with η but keeps a nonzero floor
        # (like Ipopt's final μ): a vanishing μ lets the soft-bounded
        # iterate leave the box at convergence
        μref[] = if box_barrier
            _box_barrier_μ(
                μref[],
                x,
                submodel0.box_min,
                submodel0.box_max,
                δbar,
                Float64(box_barrier_scale),
            )
        else
            0.0
        end
        # Ipopt-style restoration on stall: when the violation has not
        # improved by more than `restore_tol` over the last `restore_stall`
        # outer iterations, solve the feasibility-only problem to recover a
        # feasible reference point, re-seed the equality multipliers there
        # and continue the main phase from it.
        if restore &&
            restore_stall > 0 &&
            length(vhist) == restore_stall &&
            (
                if restore_tol_mode === :relative
                    vhist[end] >= (1 - restore_tol) * vhist[1]
                else
                    vhist[end] >= vhist[1] - restore_tol
                end
            ) &&
            n_restores < 2
            xr = _restore_feasible(
                eq_fs, ineq_fs, model, relaxed, submodel0, copy(x), restore_prox
            )
            vraw_r = _raw_violation(model, eq_rel, ineq_rel, sets_rel, xr)
            if vraw_r < _raw_violation(model, eq_rel, ineq_rel, sets_rel, x)
                x = copy(xr)
                vnorm = norm(
                    _scaled_violation(
                        eq_fs, ineq_fs, sd_rel, sets_rel, model, x, set_scales
                    ),
                    Inf,
                )
                fx = obj(x)
                if lsq_multipliers && !isempty(eq_rel)
                    y = _lsq_multipliers_scaled(scaled_obj, eq_fs, x)
                end
                if vraw_r < vraw_rest
                    x_rest = copy(xr)
                    fx_rest = fx
                    vraw_rest = vraw_r
                end
                n_restores += 1
            end
            empty!(vhist)
        end
        sd_terms = _sd_terms(model, sd_rel, ρ, x, Y, penalty)
        set_terms = _set_terms_closure(model, sets_rel, ρ, x, Msets, penalty, set_scales)
        if !isempty(sets_rel)
            sd_terms0 = sd_terms
            sd_terms = z -> sd_terms0(z) + set_terms(z)
        end
        build_al =
            sd_t -> if use_elastic
                _elastic_objective(
                    scaled_obj,
                    eq_fs,
                    ineq_fs,
                    w_eq,
                    w_ineq,
                    ρ,
                    y,
                    λ,
                    penalty,
                    sd_t,
                    elastic_mu,
                    n,
                    m_eq,
                )
            else
                _auglag_objective_scaled(
                    model,
                    relaxed,
                    (w_eq, w_ineq),
                    ρ,
                    y,
                    λ,
                    sd_t,
                    penalty,
                    scaled_obj,
                    eq_fs,
                    ineq_fs,
                    b_ineq,
                )
            end
        al_obj =
            if analytical_gradient &&
                penalty === :quadratic &&
                !use_elastic &&
                sub_alg === nothing &&
                options.curvature !== :newton
                # analytical Jᵀv-based subproblem gradient (issue #32): the
                # rrule on AnalyticalALObjective supplies the assembled
                # gradient, and the prox / box-barrier / hard-penalty
                # wrappers compose through it.
                AnalyticalALObjective(
                    scaled_obj, eq_fs, ineq_fs, y, λ, w_eq, w_ineq, Float64(ρ), sd_terms
                )
            else
                build_al(sd_terms)
            end
        # The built-in quasi-Newton solver handles only box-constrained
        # subproblems: kept (non-relaxed) constraints are folded into its
        # objective as a smooth quadratic penalty with weight hard_rho
        # (the ALM-for-relaxed / penalty-for-kept split). External
        # sub-solvers receive the kept constraints natively instead.
        use_hard_penalty =
            sub_alg === nothing && (
                !isempty(submodel0.eq_constraints.fs) ||
                !isempty(submodel0.ineq_constraints.fs)
            )
        no_eq = VectorOfFunctions(NonconvexCore.EqConstraint[])
        no_ineq = VectorOfFunctions(NonconvexCore.IneqConstraint[])
        make_submodel =
            o -> begin
                wo = if use_hard_penalty
                    _hard_penalty_objective(
                        o,
                        submodel0.eq_constraints,
                        submodel0.ineq_constraints,
                        Float64(hard_rho),
                        n,
                    )
                else
                    o
                end
                if use_elastic
                    VecModel(
                        Objective(wo),
                        use_hard_penalty ? no_eq : submodel0.eq_constraints,
                        use_hard_penalty ? no_ineq : submodel0.ineq_constraints,
                        submodel0.sd_constraints,
                        submodel0.set_constraints,
                        submodel0.set_variables,
                        [submodel0.box_min; fill(-1e3, m_eq)],
                        [submodel0.box_max; fill(1e3, m_eq)],
                        [submodel0.init; zeros(m_eq)],
                        [submodel0.integer; falses(m_eq)],
                    )
                elseif box_barrier
                    # Ipopt-style box barrier with hard projected bounds: the
                    # barrier smooths the merit near the boundary while the
                    # projection enforces it (fraction-to-boundary at tau = 1), so
                    # the iterate can never leave the box - no soft-bounds bias.
                    VecModel(
                        Objective(
                            _box_barrier_objective(
                                wo, submodel0.box_min, submodel0.box_max, μref[], δbar
                            ),
                        ),
                        use_hard_penalty ? no_eq : submodel0.eq_constraints,
                        use_hard_penalty ? no_ineq : submodel0.ineq_constraints,
                        submodel0.sd_constraints,
                        submodel0.set_constraints,
                        submodel0.set_variables,
                        submodel0.box_min,
                        submodel0.box_max,
                        submodel0.init,
                        submodel0.integer,
                    )
                else
                    VecModel(
                        Objective(wo),
                        use_hard_penalty ? no_eq : submodel0.eq_constraints,
                        use_hard_penalty ? no_ineq : submodel0.ineq_constraints,
                        submodel0.sd_constraints,
                        submodel0.set_constraints,
                        submodel0.set_variables,
                        submodel0.box_min,
                        submodel0.box_max,
                        submodel0.init,
                        submodel0.integer,
                    )
                end
            end
        x_sub = use_elastic ? [x; zeros(m_eq)] : x
        al_sub = if prox > 0
            # proximal regularization (Rockafellar): conditions the
            # subproblems of the escalated penalty
            z -> al_obj(z) + (prox / 2) * sum(abs2, z .- x_sub)
        else
            al_obj
        end
        # the finite-difference-gradient curvature reads the current AL
        # objective through this ref
        al_obj_ref[] = al_sub
        submodel = make_submodel(al_sub)
        checksubproblem!(
            submodel,
            if sub_alg === nothing
                QuasiNewtonAlg(qn_approx::AbstractHessianApprox)
            else
                sub_alg
            end;
            context="the augmented Lagrangian subproblem",
            hint="relax the constraint in `RelaxConstraintSelection` or choose a sub-solver that supports it.",
        )
        result_i = if sub_alg === nothing
            # Stall detection: if the first solve does not decrease the AL
            # merit, retry once with a 4x budget and a relaxed tolerance.
            m_old = al_obj(x_sub)
            r1 = optimize(
                submodel,
                QuasiNewtonAlg(
                    qn_approx::AbstractHessianApprox;
                    max_iter=qn_base_maxiter,
                    tol=sub_tol,
                    globalization=qn_globalization,
                    reset=qn_reset,
                    levenberg=levenberg,
                    nonmonotone=qn_nonmonotone,
                    soc=qn_soc,
                    projected_stop=qn_projected_stop,
                    cg_tol=inexact_cg ? sub_tol : nothing,
                    cg_maxiter=qn_cg_maxiter,
                    precond=qn_precond,
                ),
                x_sub,
            )
            m1 = al_obj(r1.minimizer)
            if stall_retry && !(isfinite(m1) && m1 < m_old - 1e-12 * max(1.0, abs(m_old)))
                r2 = optimize(
                    submodel,
                    QuasiNewtonAlg(
                        qn_approx::AbstractHessianApprox;
                        max_iter=4 * qn_base_maxiter,
                        tol=10 * sub_tol,
                        globalization=qn_globalization,
                        reset=qn_reset,
                        levenberg=levenberg,
                        nonmonotone=qn_nonmonotone,
                        soc=qn_soc,
                        projected_stop=qn_projected_stop,
                        cg_tol=inexact_cg ? 10 * sub_tol : nothing,
                        cg_maxiter=qn_cg_maxiter,
                        precond=qn_precond,
                    ),
                    x_sub,
                )
                m2 = al_obj(r2.minimizer)
                if isfinite(m2) && m2 < m1
                    r1 = r2
                end
            end
            r1
        else
            lh = if je_plain === nothing
                nothing
            else
                (z, σ, lam) -> begin
                    H = _jump_al_hessian(
                        je_plain,
                        z,
                        eq_fs,
                        ineq_fs,
                        scales,
                        y,
                        λ,
                        w_eq,
                        w_ineq,
                        Ref(ρ),
                        fscale,
                        box_barrier ? submodel0.box_min : nothing,
                        box_barrier ? submodel0.box_max : nothing,
                        μref,
                        δbar,
                    )
                    prox > 0 && (H[diagind(H)] .+= prox)
                    σ .* H
                end
            end
            sub_opts_i = if options.sub_options === nothing
                nothing
            else
                _set_sub_tol(sub_alg, options.sub_options, sub_tol)
            end
            if sub_opts_i === nothing
                if lh === nothing
                    optimize(submodel, sub_alg, x_sub)
                else
                    optimize(submodel, sub_alg, x_sub; lagrangian_hessian=lh)
                end
            elseif lh === nothing
                optimize(submodel, sub_alg, x_sub; options=sub_opts_i)
            else
                optimize(submodel, sub_alg, x_sub; options=sub_opts_i, lagrangian_hessian=lh)
            end
        end
        x_new = copy(result_i.minimizer)
        if two_level
            # Two-level semi-proximal inner loop (SDPNAL+-style sPADMM for
            # the semidefinite block): alternate the multiplier projection
            # with prox-augmented subproblem solves AL(x) + τ/2‖x - x̂‖².
            x_two = x_new
            τ = Float64(two_level_tau)
            for _ in 1:two_level_iters
                _sd_multiplier_update!(model, sd_rel, ρ, x_two, Y, penalty)
                _set_update!(model, sets_rel, ρ, x_two, Msets, penalty)
                sd_t2 = _sd_terms(model, sd_rel, ρ, x_two, Y, penalty)
                if !isempty(sets_rel)
                    set_t2 = _set_terms_closure(
                        model, sets_rel, ρ, x_two, Msets, penalty, set_scales
                    )
                    sd_t2_0 = sd_t2
                    sd_t2 = z -> sd_t2_0(z) + set_t2(z)
                end
                al_two = build_al(sd_t2)
                prox = z -> al_two(z) + (τ / 2) * sum(abs2, z .- x_two)
                sub2 = make_submodel(prox)
                r2 = optimize(
                    sub2,
                    QuasiNewtonAlg(
                        qn_approx::AbstractHessianApprox;
                        max_iter=max(50, qn_base_maxiter ÷ 2),
                        tol=sub_tol,
                        globalization=qn_globalization,
                        reset=qn_reset,
                        levenberg=levenberg,
                        nonmonotone=qn_nonmonotone,
                        soc=qn_soc,
                        projected_stop=qn_projected_stop,
                        cg_tol=inexact_cg ? sub_tol : nothing,
                        cg_maxiter=qn_cg_maxiter,
                        precond=qn_precond,
                    ),
                    x_two,
                )
                x_two = r2.minimizer
            end
            x_new = x_two
        end
        # Filter-based outer globalization: backtrack along the subsolver
        # direction until the (merit, violation) pair is not dominated by
        # any previous filter entry.
        if filter
            γ = 1e-4
        end
        _filter_accept(θ, φ) =
            !filter || all(
                θ <= filter_f[i] - γ * filter_v[i] || φ <= filter_v[i] - γ * filter_f[i] for
                i in eachindex(filter_f)
            )
        d = x_new .- x_sub
        α = 1.0
        accepted = false
        mt = al_obj(x_sub)
        # Objective recovery ingredients (classical quadratic mode, no
        # elastic, no sd): the AL value and the raw scaled constraint
        # values at the accepted point recover f without re-evaluating
        # anything (Percival's objective recovery, and it also removes the
        # duplicate constraint evaluation of the violation recomputation).
        can_recover =
            penalty === :quadratic && !use_elastic && isempty(sd_rel) && isempty(sets_rel)
        eqvals_acc = nothing
        ineqvals_acc = nothing
        mt_acc = mt
        φt_acc = Inf
        for _ in 1:10
            xt = x_sub .+ α .* d
            mt = al_obj(xt)
            xt_orig = use_elastic ? @view(xt[1:n]) : xt
            eqvals, ineqvals = _scaled_constraint_values(eq_fs, ineq_fs, xt_orig)
            vvals = _scaled_violation_from(eqvals, ineqvals)
            for i in sd_rel
                append!(vvals, _sd_violation(model, i, xt_orig))
            end
            for i in sets_rel
                append!(vvals, _set_violation(model, i, xt_orig))
            end
            φt = norm(vvals, Inf)
            if isfinite(mt) &&
                isfinite(φt) &&
                _filter_accept(mt, φt) &&
                (!watchdog || mt <= watchdog_fbar * (1 + watchdog_tol))
                x_new = xt
                accepted = true
                if can_recover
                    eqvals_acc = eqvals
                    ineqvals_acc = ineqvals
                    mt_acc = mt
                    φt_acc = φt
                end
                break
            end
            α *= 0.5
        end
        if watchdog && accepted
            # Recursive-mean update of the nonmonotone reference (BM style);
            # left unchanged on rejection so persistent failures escalate.
            watchdog_fbar = if isfinite(watchdog_fbar)
                (1 - watchdog_tau) * watchdog_fbar + watchdog_tau * mt
            else
                mt
            end
        end
        if !accepted && filter
            # No acceptable step: record the current point in the filter and
            # keep it (the outer update will escalate).
            push!(filter_f, al_obj(x_sub))
            push!(
                filter_v,
                norm(
                    _scaled_violation(
                        eq_fs, ineq_fs, sd_rel, sets_rel, model, x, set_scales
                    ),
                    Inf,
                ),
            )
        end
        if watchdog && !accepted
            # Watchdog rejection: discard the step entirely and let the outer
            # update escalate the penalty (the filter alone records and keeps
            # the step; the watchdog does not — it is the safety net against
            # subsolvers that return merit-increasing steps).
            x_new = x_sub
        end
        x = use_elastic ? x_new[1:n] : x_new
        vnorm = if accepted && can_recover
            φt_acc
        else
            norm(_scaled_violation(eq_fs, ineq_fs, sd_rel, sets_rel, model, x, set_scales), Inf)
        end
        f_new = if accepted && can_recover
            _recover_objective(mt_acc, eqvals_acc, ineqvals_acc, y, λ, w_eq, w_ineq, ρ, fscale)
        else
            obj(x)
        end
        ok = isfinite(vnorm) && isfinite(f_new) && vnorm < 1e3 * max(vprev, 1.0)
        if !ok
            # The subproblem blew up (typical of an over-aggressive penalty):
            # discard the iterate, escalate the penalty and retry. The
            # restoration phase (opt-in) recovers a feasibility-improved
            # point instead of just reverting.
            if restore
                # restore from the last good iterate (x may hold the
                # blown-up point here)
                xgood = use_elastic ? @view(x_sub[1:n]) : x_sub
                x = _restore_feasible(
                    eq_fs, ineq_fs, model, relaxed, submodel0, copy(xgood), restore_prox
                )
                vnorm = norm(
                    _scaled_violation(
                        eq_fs, ineq_fs, sd_rel, sets_rel, model, x, set_scales
                    ),
                    Inf,
                )
                f_new = obj(x)
            else
                x = _failed_iterate_fallback(trajectory, x_init)
                vnorm = vprev
                f_new = fx
            end
        end
        fx = f_new
        push!(trajectory, (copy(x), fx, vnorm))
        # Optional per-iteration callback: receives (iter, x, f(x), vnorm)
        # with a copy of the iterate; returning true stops the loop (the
        # reported result is the iterate just recorded).
        if on_iteration !== nothing
            on_iteration(iter, copy(x), fx, vnorm) && break
        end

        ρ_before = ρ
        if vnorm <= η || vnorm <= ω * vprev
            # Multiplier updates (the practical ALM condition: update whenever
            # feasibility improved sufficiently, not only once η is reached —
            # this stops the penalty from escalating unnecessarily).
            # Eckstein-style adaptive relaxation of the multiplier step: when
            # the subproblem was solved less accurately than requested (its
            # stationarity residual at the accepted iterate exceeds the
            # requested tolerance), the dual step is scaled down by the
            # achieved-accuracy ratio, so a loose inner solve cannot poison
            # the multiplier estimates. The residual is the projected
            # gradient of the subproblem objective (the same metric the
            # built-in quasi-Newton solver uses to stop); the barrier/prox
            # corrections are ignored (cheap approximation).
            θ_mult = 1.0
            if multiplier_relax
                gal = NonconvexCore.value_gradient(al_sub, x_new)[2]
                galn = use_elastic ? @view(gal[1:n]) : gal
                xn = use_elastic ? @view(x_new[1:n]) : x_new
                pg = _projected_gnorm(galn, xn, submodel0.box_min, submodel0.box_max)
                θ_mult = clamp(sub_tol / max(pg, sub_tol), 0.1, 1.0)
            end
            if dual_secant && !use_elastic && !isempty(eq_rel)
                # Broyden secant-corrected equality multiplier update (the
                # Birgin-Martinez "improving ultimate convergence" device):
                # C is the inverse secant model of the residual map of
                # y += theta*rho*w*c; the step becomes the quasi-Newton
                # step -C*r (the first step reproduces the plain ascent,
                # C = -I) with the good-Broyden inverse update from the
                # secant pairs of the map, safeguarded to the plain ascent
                # when the model step exceeds twice its norm. Replaces the
                # plain update and the equality Anderson mixing.
                cflat = Float64[]
                for k in eachindex(eq_rel)
                    append!(cflat, θ_mult .* (ρ * w_eq[k]) .* _vecval(eq_fs[k](x)))
                end
                yflat = reduce(vcat, y; init=Float64[])
                if bdual[] === nothing
                    _dual_secant_init!(bdual, length(yflat))
                    step = cflat
                else
                    if ysec_prev !== nothing
                        _dual_secant_update!(
                            bdual[], yflat .- ysec_prev, cflat .- csec_prev
                        )
                    end
                    step = .-(bdual[] * cflat)
                    norm(step) <= 2 * norm(cflat) || (step = cflat)
                end
                ysec_prev = yflat
                csec_prev = cflat
                off = 0
                for k in eachindex(eq_rel)
                    mk = length(y[k])
                    y[k] .= clamp.(
                        yflat[(off + 1):(off + mk)] .+ step[(off + 1):(off + mk)], -1e8, 1e8
                    )
                    off += mk
                end
            else
                if use_elastic
                    s = x_new[(n + 1):end]
                    off = 0
                    for k in eachindex(eq_rel)
                        ck = _vecval(eq_fs[k](x))
                        sk = s[(off + 1):(off + length(ck))]
                        y[k] .= y[k] .+ θ_mult .* (ρ * w_eq[k]) .* (ck .- sk)
                        off += length(ck)
                    end
                else
                    for k in eachindex(eq_rel)
                        y[k] .= y[k] .+ θ_mult .* (ρ * w_eq[k]) .* _vecval(eq_fs[k](x))
                    end
                end
                # Anderson acceleration of the equality multiplier updates.
                # The residual must match the fixed-point map actually being
                # iterated: in elastic mode the update is y += ρw(c − s), so the
                # slack `s` belongs in the residual too (using ρw·c accelerated
                # the wrong map).
                if !isempty(eq_rel)
                    y_new_flat = reduce(vcat, y; init=Float64[])
                    resid = Float64[]
                    if use_elastic
                        s = x_new[(n + 1):end]
                        off = 0
                        for k in eachindex(eq_rel)
                            ck = _vecval(eq_fs[k](x))
                            sk = s[(off + 1):(off + length(ck))]
                            append!(resid, θ_mult .* (ρ * w_eq[k]) .* (ck .- sk))
                            off += length(ck)
                        end
                    else
                        for k in eachindex(eq_rel)
                            append!(resid, θ_mult .* (ρ * w_eq[k]) .* _vecval(eq_fs[k](x)))
                        end
                    end
                    y_prev_flat = y_new_flat .- resid
                    push!(and_history, (y_prev_flat, resid))
                    length(and_history) > anderson_depth && popfirst!(and_history)
                    y_mixed = _anderson_mix(and_history, y_new_flat)
                    off = 0
                    for k in eachindex(eq_rel)
                        mk = length(y[k])
                        y[k] .= if safeguard
                            clamp.(y_mixed[(off + 1):(off + mk)], -1e8, 1e8)
                        else
                            y_mixed[(off + 1):(off + mk)]
                        end
                        off += mk
                    end
                end
                for k in eachindex(eq_rel)
                    safeguard && clamp!(y[k], -1e8, 1e8)
                end
            end
            k = 1
            # flat buffer of the unprojected inequality increments ρw·g
            # (stashed here so the Anderson mixing below does not
            # re-evaluate the constraint functions)
            ineq_resid = Float64[]
            for i in eachindex(ineq_rel)
                g = _vecval(ineq_fs[i](x))
                w = w_ineq[k]
                cbar = pbm_adaptive ? b_ineq[k] : ρ * w
                if penalty in (:pbm, :mixed, :mixed_logdet)
                    feas = g .<= 0
                    λ[k][feas] .= λ[k][feas] ./ (1 .+ λ[k][feas] .* g[feas] ./ cbar)
                    λ[k][.!feas] .= max.(λ[k][.!feas] .+ (ρ * w) .* g[.!feas], 0.0)
                else
                    λ[k] .= max.(λ[k] .+ θ_mult .* (ρ * w) .* g, 0.0)
                end
                anderson_ineq &&
                    penalty === :quadratic &&
                    append!(ineq_resid, θ_mult .* (ρ * w) .* g)
                safeguard && clamp!(λ[k], 0.0, 1e8)
                if pbm_adaptive && penalty in (:pbm, :mixed, :mixed_logdet)
                    # Ipopt-style barrier adaptation: the barrier coefficient
                    # tracks the complementarity of the implied duals on the
                    # feasible side (grow when far from complementary, shrink
                    # when well-complemented).
                    feas = g .<= 0
                    if any(feas)
                        gf = λ[k][feas] .* g[feas]
                        θ = sum(abs, gf) / length(gf)
                        b_ineq[k] = clamp(
                            b_ineq[k] * clamp(1e-2 / max(θ, 1e-12), 0.5, 2.0), 1e-2, ρ * w
                        )
                    end
                end
                k += 1
            end
            if anderson_ineq && penalty === :quadratic && !isempty(ineq_resid)
                # Anderson acceleration of the inequality multiplier map (the
                # classical quadratic penalty): the depth-2 least-squares mix
                # of the unprojected iterates λ + ρw·g, re-projected onto the
                # nonnegative orthant (ADMM-style Anderson with re-projection).
                λ_new_flat = reduce(vcat, λ; init=Float64[])
                λ_prev_flat = λ_new_flat .- ineq_resid
                push!(and_ineq_history, (λ_prev_flat, ineq_resid))
                length(and_ineq_history) > anderson_depth && popfirst!(and_ineq_history)
                λ_mixed = _anderson_mix(and_ineq_history, λ_new_flat)
                off = 0
                for i in eachindex(ineq_rel)
                    mk = length(λ[i])
                    λ[i] .= if safeguard
                        clamp.(λ_mixed[(off + 1):(off + mk)], 0.0, 1e8)
                    else
                        max.(λ_mixed[(off + 1):(off + mk)], 0.0)
                    end
                    off += mk
                end
            end
            if spectral_rho && !use_elastic && (!isempty(eq_rel) || !isempty(ineq_rel))
                # Spectral (Barzilai-Borwein) penalty update on the dual map
                # (Xu-Figueiredo-Goldstein style, ADMM-flavored): the dual
                # step 1/rho is estimated from the last two iterates as
                # rho ~ ||dc||^2 / <dc, dy> with dc the change in the
                # constraint values and dy the change in the multipliers.
                # Safeguards: only a credible positive correlation updates
                # rho, the change is limited to [rho/rho_decr, rho*rho_incr]
                # and the result to [rho_init/10, rho_max]. Replaces the
                # two-way decrease on good steps (it subsumes it). The
                # semidefinite multipliers (matrices) are not part of the
                # flat vectors; the elastic mode is not supported.
                cflat = Float64[]
                for k in eachindex(eq_rel)
                    append!(cflat, _vecval(eq_fs[k](x)))
                end
                for i in eachindex(ineq_rel)
                    append!(cflat, _vecval(ineq_fs[i](x)))
                end
                yflat = reduce(vcat, vcat(y, λ); init=Float64[])
                if cflat_prev !== nothing && yflat_prev !== nothing
                    Δc = cflat .- cflat_prev
                    Δy = yflat .- yflat_prev
                    num = dot(Δc, Δc)
                    den = dot(Δc, Δy)
                    if num > 0 && den > 0
                        ρ_spec = clamp(
                            num / den,
                            ρ / Float64(options.rho_decr),
                            ρ * Float64(options.rho_incr),
                        )
                        ρ = clamp(
                            ρ_spec, Float64(options.rho_init) / 10, Float64(options.rho_max)
                        )
                    end
                end
                cflat_prev = cflat
                yflat_prev = yflat
            end
            _sd_multiplier_update!(model, sd_rel, ρ, x, Y, penalty)
            _set_update!(model, sets_rel, ρ, x, Msets, penalty)
            if outer_schedule === :bm
                # Birgin-Martinez complexity scaling: η, ω tighten with 1/ρ.
                η = max(η / ρ, ctol)
                ω = max(ω / ρ, sqrt(eps(Float64)))
            else
                η = max(η / ρ^0.9, ctol)
                ω = max(ω / ρ^0.1, sqrt(eps(Float64)))
            end
            # Two-way update (Curtis, Jiang & Robinson): once feasibility is
            # nearly attained, back the penalty down so the subproblems stay
            # well conditioned. The spectral update (when enabled) subsumes
            # this on good steps.
            spectral_rho || (ρ = _two_way_step(ρ, options, two_way, vnorm, η))
        else
            # Selective penalty update: escalate only the most-violated
            # constraint groups instead of the global penalty.
            ρ, η, ω = _escalate_penalty!(
                ρ, η, ω, w_eq, w_ineq, w_max, options, eq_rel, ineq_rel, eq_fs, ineq_fs, x
            )
        end
        vprev = vnorm
        if restore && restore_stall > 0
            push!(vhist, vnorm)
            length(vhist) > restore_stall && popfirst!(vhist)
        end
        if watchdog && ρ != ρ_before
            # The AL objective's scale changed with the penalty: the frozen
            # reference no longer applies, so give the next step a free pass
            # and re-anchor the running mean.
            watchdog_fbar = Inf
        end
        jump_newton_active && (ρref[] = ρ)
        if jump_newton_active && qn_approx isa JumpNewtonTRApprox
            # Warm-start the Steihaug preconditioner with the Cholesky of
            # the AL Hessian at the accepted iterate: it is stale across
            # outer iterations (multipliers and ρ change), but it is a
            # strong preconditioner for the ill-conditioned subproblems.
            Hc = qn_approx.hessfun(x)
            qn_approx.precond = try
                cholesky(Hermitian(Hc + 1e-8 * I); check=false)
            catch
                nothing
            end
        end
        # KKT-complete stopping (opt-in): on top of feasibility, require the
        # stationarity and complementarity residuals to be small — the
        # box-projected gradient of the scaled Lagrangian at the current
        # multipliers, and max |λⱼgⱼ| over the relaxed inequalities.
        stat_ok = true
        comp_ok = true
        if stop_kkt && vnorm <= ctol
            _, ∇f = NonconvexCore.value_gradient(scaled_obj, x)
            g = copy(∇f)
            for (k, c) in enumerate(eq_fs)
                gJ = NonconvexCore.value_gradient(z -> dot(_vecval(c(z)), y[k]), x)[2]
                gJ === nothing || (g .+= gJ)
            end
            for (k, c) in enumerate(ineq_fs)
                gJ = NonconvexCore.value_gradient(z -> dot(_vecval(c(z)), λ[k]), x)[2]
                gJ === nothing || (g .+= gJ)
            end
            # The set-constraint multipliers: ⟨y, g_eq⟩ + ⟨λ, g_ineq⁺⟩ +
            # Σₖ⟨Yₖ, Pₖ⟩ (the PSD part batched into one rule'd call).
            for (k, i) in enumerate(sets_rel)
                gJ = NonconvexCore.value_gradient(
                    z -> _set_multiplier_terms(model, i, z, Msets[k], set_scales[k]), x
                )[2]
                gJ === nothing || (g .+= gJ)
            end
            pg = _projected_gnorm(g, x, submodel0.box_min, submodel0.box_max)
            comp = 0.0
            for k in eachindex(ineq_rel)
                comp = max(comp, maximum(abs.(λ[k] .* _vecval(ineq_fs[k](x)))))
            end
            for (k, i) in enumerate(sets_rel)
                comp = max(comp, _set_complementarity(model, i, x, Msets[k], set_scales[k]))
            end
            stat_ok = pg <= kkt_tol
            comp_ok = comp <= kkt_tol
        end
        # Infeasibility detection (Percival's dual certificate): once the
        # penalty is capped and the iterate is still infeasible, a
        # vanishing unit-weight penalty-part gradient ∇(½Σ‖c̃‖² + ½Σ‖g̃⁺‖²)
        # certifies — up to the tolerances — that no feasible point is
        # reachable from the relaxation.
        if !infeasible_flag &&
            vnorm > ctol &&
            ρ >= Float64(options.rho_max) &&
            penalty === :quadratic &&
            !use_elastic &&
            (!isempty(eq_rel) || !isempty(ineq_rel))
            gp = _penalty_gradient(
                eq_fs, ineq_fs, ones(length(eq_fs)), ones(length(ineq_fs)), 1.0
            )(
                x
            )
            pgnorm = norm(gp)
            if pgnorm < sqrt(ctol) * vnorm
                infeasible_flag = true
                break
            end
        end
        vnorm <= ctol && stat_ok && comp_ok && break
    end
    # Report the best restoration point when it is strictly more feasible
    # (in the raw metric the restoration minimizes) than the final
    # main-phase iterate — the main phase is not feasibility-preserving,
    # so its terminal iterate can be infeasible even though a restoration
    # phase reached feasibility earlier.
    vraw_final = _raw_violation(model, eq_rel, ineq_rel, sets_rel, x)
    reported_x = (x_rest !== nothing && vraw_rest < vraw_final) ? x_rest : x
    reported_fx = (x_rest !== nothing && vraw_rest < vraw_final) ? fx_rest : fx
    workspace.state = (
        n=n,
        m_eq=m_eq,
        neq=length(eq_fs),
        nineq=length(ineq_fs),
        eq_rel=copy(eq_rel),
        ineq_rel=copy(ineq_rel),
        sd_rel=copy(sd_rel),
        elastic=use_elastic,
        x=copy(reported_x),
        y=[copy(yk) for yk in y],
        λ=[copy(λk) for λk in λ],
        scales=copy(scales),
        w_eq=copy(w_eq),
        w_ineq=copy(w_ineq),
        ρ=ρ,
        η=η,
        ω=ω,
        and_history=copy(and_history),
        and_ineq_history=copy(and_ineq_history),
        filter_f=copy(filter_f),
        filter_v=copy(filter_v),
        approx=qn_approx,
    )
    return AugLagResult(
        copy(reported_x),
        reported_fx,
        nothing,
        keep_all ? trajectory : nothing,
        infeasible_flag ? :infeasible : :unknown,
    )
end

# Least-squares multiplier initialization over the scaled equality
# functions, solved matrix-free with conjugate gradients: the algorithm only
# ever needs Jacobian-vector products (Jv and Jᵀv), never a full Jacobian.
function _cg_solve!(y, A, b; max_cg=min(length(b), 50), rtol=1e-2)
    r = b .- A(y)
    p = copy(r)
    rsold = dot(r, r)
    iszero(rsold) && (y.=0; return y)
    for _ in 1:max_cg
        Ap = A(p)
        α = rsold / dot(p, Ap)
        y .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)
        rsnew <= rtol^2 * dot(b, b) && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return y
end

function _lsq_multipliers_scaled(obj, eq_fs, x)
    _, ∇f = NonconvexCore.value_gradient(obj, x)
    ∇f = copy(∇f)
    y = Vector{Vector{Float64}}()
    ε = sqrt(eps(Float64)) * (1 + norm(x))
    for c in eq_fs
        c0 = _vecval(c(x))
        m = length(c0)
        # Jv: one finite-difference directional derivative per CG iteration
        # (this is a warm start, so FD precision is more than adequate and no
        # full Jacobian is ever formed).
        Jv(v) =
            (_vecval(c(x .+ (ε / max(norm(v), 1e-12)) .* v)) .- c0) ./
            (ε / max(norm(v), 1e-12))
        # Jᵀv: one reverse pass (adjoint product).
        Jtv(v) = NonconvexCore.value_gradient(x -> dot(_vecval(c(x)), v), x)[2]
        b = -Jv(∇f)
        A(y) = Jv(Jtv(y)) .+ 1e-8 .* y
        yk = zeros(m)
        _cg_solve!(yk, A, b)
        push!(y, yk)
    end
    return y
end

# Elastic augmented Lagrangian objective on z = [x; s]: the equality groups
# get slack variables with an l1-like (pseudo-Huber) penalty.
_huber(t, ε) = sqrt(t^2 + ε^2)
function _elastic_objective(
    scaled_obj,
    eq_fs,
    ineq_fs,
    w_eq,
    w_ineq,
    ρ::Real,
    y,
    λ,
    penalty::Symbol,
    sd_terms::Function,
    μ::Real,
    n::Int,
    m_eq::Int,
)
    ε = 1e-4
    function _objective(z)
        x = @view z[1:n]
        s = @view z[(n + 1):end]
        target = scaled_obj(x)
        off = 0
        for (k, c) in enumerate(eq_fs)
            cv = _vecval(c(x))
            sk = s[(off + 1):(off + length(cv))]
            r = ρ * w_eq[k]
            target += dot(y[k], cv .- sk) + r / 2 * dot(cv .- sk, cv .- sk)
            for si in sk
                target += μ * _huber(si, ε)
            end
            off += length(cv)
        end
        for (k, c) in enumerate(ineq_fs)
            g = _vecval(c(x))
            r = ρ * w_ineq[k]
            if penalty in (:pbm, :mixed, :mixed_logdet)
                feas = g .<= 0
                target += _pbm_barrier(g[feas], λ[k][feas], r)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    target += dot(λ[k], gp) + r / 2 * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                target += dot(λ[k], gp) + r / 2 * dot(gp, gp)
            end
        end
        target += sd_terms(x)
        return target
    end
    return _objective
end

# Extension hooks for semidefinite constraints (see ext/NonconvexAugLagLabSDPExt.jl)
function _sd_init(model, sd_rel, ρ)
    isempty(sd_rel) && return nothing
    return _sd_violation(model, first(sd_rel), model.init) # throws when the extension is missing
end
function _sd_terms(model, sd_rel, ρ, x, Y, penalty)
    return x -> 0.0
end
_sd_multiplier_update!(model, sd_rel, ρ, x, Y, penalty) = nothing

# ------------------------------------------------------------------
# Jump (JuMP-converted) Newton trust region: the exact Hessian of the
# quadratic augmented Lagrangian objective comes from the MathOptInterface
# evaluator instead of AD (Zygote cannot differentiate the evaluator rrules
# a second time).
# ------------------------------------------------------------------
mutable struct JumpNewtonTRApprox <: AbstractHessianApprox
    hessfun::Function
    n::Int
    H::Matrix{Float64}
    precond::Any          # stale Cholesky warm-start for the Steihaug CG
end
JumpNewtonTRApprox(hessfun) = JumpNewtonTRApprox(hessfun, 0, zeros(0, 0), nothing)
function init!(H::JumpNewtonTRApprox, n::Int, x0)
    H.n = n
    H.H = zeros(n, n)
    H.precond = nothing
    return H
end
reset!(H::JumpNewtonTRApprox) = H
update!(H::JumpNewtonTRApprox, s, y) = H
function apply!(Hv, H::JumpNewtonTRApprox, v)
    return mul!(Hv, H.H, v)
end
solve!(d, H::JumpNewtonTRApprox, g) = copyto!(d, g)
_is_newton(::JumpNewtonTRApprox) = true

# Unwrap the JuMPEvaluator behind a JuMP-converted VecModel objective, or
# `nothing` when the model is not JuMP-converted.
function _jump_evaluator(obj)
    raw = obj isa Objective ? NonconvexCore.getfunction(obj) : obj
    raw = raw isa NonconvexCore.FlattenedFunction ? NonconvexCore.getrawfunction(raw) : raw
    raw isa NonconvexCore.JuMPModelObjective || return nothing
    return raw.m
end

# Whether a constraint block belongs to the given JuMP evaluator (one of the
# JuMP-converted model's own blocks, not a composite of several).
function _is_jump_block(c, je)
    raw = c isa NonconvexCore.AbstractConstraint ? NonconvexCore.getfunction(c) : c
    raw = raw isa NonconvexCore.FlattenedFunction ? NonconvexCore.getrawfunction(raw) : raw
    raw = raw isa NonconvexCore.FunctionWrapper ? NonconvexCore.getfunction(raw) : raw
    return (raw isa NonconvexCore.JuMPModelIneq || raw isa NonconvexCore.JuMPModelEq) &&
           raw.m === je
end

# ------------------------------------------------------------------
# Box barrier (Ipopt-style): the subproblem's finite box bounds enter as a
# log barrier with a C² extension outside, making the subproblem smooth and
# unconstrained; μ relaxes with the outer feasibility target η.
# ------------------------------------------------------------------
# Smooth C² extension of log(t) below the thickness δ: matches the value,
# gradient and Hessian of log at t = δ, and stays defined (quadratic) for
# t ≤ 0, so the barrier objective is smooth everywhere.
function _box_barrier_term(t, δ)
    if t >= δ
        return log(t)
    else
        return log(δ) + (t - δ) / δ - (t - δ)^2 / (2δ^2)
    end
end
# Hessian of the barrier term: 1/t² for t ≥ δ and 1/δ² below.
_box_barrier_curv(t, δ) = inv(max(t, δ)^2)

# Ipopt-style barrier adaptation for the box barrier: μ tracks the mean
# implied dual μ/dist on the near-bound coordinates (grow when the duals
# are weak, shrink when well-established), clamped to [1e-4, 10].
function _box_barrier_μ(μ, x, l, u, δ, scale)
    s = 0.0
    k = 0
    for i in eachindex(x)
        if isfinite(l[i]) && x[i] - l[i] < 10δ
            s += μ / max(x[i] - l[i], δ)
            k += 1
        end
        if isfinite(u[i]) && u[i] - x[i] < 10δ
            s += μ / max(u[i] - x[i], δ)
            k += 1
        end
    end
    κ = k == 0 ? 0.0 : s / k
    if k == 0
        # no coordinate near a bound: the barrier is irrelevant, decay μ
        return max(μ / 2, 1e-4)
    end
    return clamp(μ * clamp(1e-2 / max(κ, 1e-12), 0.5, 2.0), 1e-4, 10.0 * scale)
end

function _box_barrier_objective(f, l, u, μ, δ)
    return x -> begin
        val = f(x)
        for i in eachindex(x)
            isfinite(l[i]) && (val -= μ * _box_barrier_term(x[i] - l[i], δ))
            isfinite(u[i]) && (val -= μ * _box_barrier_term(u[i] - x[i], δ))
        end
        return val
    end
end

# Exact dense Hessian of the (scaled) quadratic augmented Lagrangian
# objective at `x`, assembled from the evaluator's Lagrangian Hessian and
# the exact Jacobian:
#     H = ∇²f/fscale + Σ μᵢ ∇²c̃ᵢ + ρ Σᵢ(active) wᵢ ∇c̃ᵢ ∇c̃ᵢᵀ
# with c̃ = c/s the scaled constraint functions, μ = (multiplier + ρ w c̃)/s,
# and the ρ JᵀJ term restricted to the active (violated) inequality rows.
function _jump_al_hessian(
    je,
    x,
    eq_fs,
    ineq_fs,
    scales,
    y,
    λ,
    w_eq,
    w_ineq,
    ρref,
    fscale,
    l=nothing,
    u=nothing,
    μref=Ref(0.0),
    δ=1e-4,
)
    n = length(x)
    H = zeros(n, n)
    neq = length(eq_fs)
    # multipliers in the evaluator's converted row order: inequalities first
    μ = zeros(je.nineq + je.neq)
    off = 0
    for k in eachindex(ineq_fs)
        g = _vecval(ineq_fs[k](x))
        s = scales[neq + k]
        w = w_ineq[k]
        for i in eachindex(g)
            μ[off + i] = g[i] > 0 ? (λ[k][i] + ρref[] * w * g[i]) / s : 0.0
        end
        off += length(g)
    end
    for k in eachindex(eq_fs)
        c = _vecval(eq_fs[k](x))
        s = scales[k]
        w = w_eq[k]
        for i in eachindex(c)
            μ[off + i] = (y[k][i] + ρref[] * w * c[i]) / s
        end
        off += length(c)
    end
    sv = zeros(length(je.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(je, sv, x, 1.0 / fscale, μ)
    @inbounds for k in eachindex(je.hess_rows)
        i = je.hess_rows[k]
        j = je.hess_cols[k]
        H[i, j] += sv[k]
        i == j || (H[j, i] += sv[k])
    end
    # ρ JᵀJ: exact Jacobian rows scaled by sqrt(ρ w)/s, inactive rows zeroed
    J = NonconvexCore.jump_jacobian(je, x)
    D = zeros(size(J, 1))
    off = 0
    for k in eachindex(ineq_fs)
        g = _vecval(ineq_fs[k](x))
        s = scales[neq + k]
        w = w_ineq[k]
        for i in eachindex(g)
            D[off + i] = g[i] > 0 ? sqrt(ρref[] * w) / s : 0.0
        end
        off += length(g)
    end
    for k in eachindex(eq_fs)
        c = _vecval(eq_fs[k](x))
        s = scales[k]
        w = w_eq[k]
        for i in eachindex(c)
            D[off + i] = sqrt(ρref[] * w) / s
        end
        off += length(c)
    end
    DJ = D .* J
    H .+= DJ' * DJ
    # box-barrier curvature (diagonal; 1/max(t, δ)² on each side)
    if l !== nothing
        μ = μref[]
        for i in 1:n
            isfinite(l[i]) && (H[i, i] += μ * _box_barrier_curv(x[i] - l[i], δ))
            isfinite(u[i]) && (H[i, i] += μ * _box_barrier_curv(u[i] - x[i], δ))
        end
    end
    return H
end
