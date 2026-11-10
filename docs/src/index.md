# Augmented Lagrangian in pure Julia

## Description

`NonconvexAugLagLab.jl` provides [`AugLagAlg`](@ref), an augmented
Lagrangian meta-algorithm for equality-, inequality- and (with
NonconvexSemidefinite loaded) semidefinite-constrained problems. Constraints
selected for relaxation ([`RelaxConstraintSelection`](@ref)) are folded into
an augmented Lagrangian objective; the remaining constraints stay hard for
the sub-solver. The outer loop follows the practical scheme of Birgin &
Martínez: feasibility-driven adaptive penalty, tolerance sequences,
multiplier safeguards and infeasibility detection. Three penalty modes are
available (`penalty = :quadratic | :pbm | :mixed`, see [Options](@ref
Options) below). The built-in sub-solver is a smooth box-constrained
quasi-Newton optimizer, [`QuasiNewtonAlg`](@ref), with pluggable Hessian
approximations.

The theory is reviewed in [Theory](theory.md): the classical ALM, the
practical Birgin–Martínez loop, the penalty-barrier-multiplier (PBM) method,
the projected ALM for semidefinite constraints, and the quasi-Newton
sub-solver machinery.

## Quick start

```julia
using NonconvexAugLagLab

# min x₁² + x₂² s.t. x₁ + x₂ = 1 → (0.5, 0.5)
m = Model(x -> sum(x .^ 2))
addvar!(m, [-5.0, -5.0], [5.0, 5.0])
add_eq_constraint!(m, x -> x[1] + x[2] - 1)
r = optimize(m, AugLagAlg(), [0.0, 0.0])
r.minimizer   # ≈ [0.5, 0.5]
r.minimum     # ≈ 0.5
```

[`AugLagAlg`](@ref)`()` uses the built-in `QuasiNewtonAlg`, which requires
every constraint to be relaxed. To keep some constraints hard, pass a
sub-solver that supports them:

```julia
using NonconvexAugLagLab, NonconvexIpopt

alg = AugLagAlg(IpoptAlg())                       # Ipopt solves the subproblems
options = AugLagOptions(
    sub_options = IpoptOptions(max_iter = 100),   # sub-solver options
    relaxed = RelaxConstraintSelection(eq = false), # keep equalities hard
)
r = optimize(m, alg, [0.0, 0.0], options = options)
```

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
| `lsq_multipliers` | `false` | initialize equality multipliers from the least-squares stationarity residual |
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

## Result

`optimize` returns an [`AugLagResult`](@ref) with the usual
`GenericResult`-style fields `minimizer` and `minimum`. With
`AugLagOptions(keep_all = true)` the full outer-loop trajectory — one
`(x, objective, violation)` tuple per outer iteration — is available as
`r.trajectory`.

```julia
options = AugLagOptions(keep_all = true)
r = optimize(m, AugLagAlg(), [0.0, 0.0], options = options)
r.trajectory  # vector of (x, f(x), ‖v(x)‖∞) per outer iteration
```

## Quasi-Newton sub-solver

[`QuasiNewtonAlg`](@ref) is a smooth box-constrained quasi-Newton optimizer
usable standalone or as the built-in sub-solver of `AugLagAlg()`. The
Hessian model is selected by passing any
[`AbstractHessianApprox`](@ref):

```julia
# L-BFGS (default): memory-limited two-loop recursion
alg = QuasiNewtonAlg()
alg = QuasiNewtonAlg(LBFGSApprox(20))

# Dense BFGS: inverse Hessian (default) or direct Hessian + factorization
alg = QuasiNewtonAlg(DenseBFGSApprox())
alg = QuasiNewtonAlg(DenseBFGSApprox(form = :hessian))

# Structured BFGS: secants restricted to a user-defined sparsity pattern
using SparseArrays
alg = QuasiNewtonAlg(StructuredBFGSApprox(spdiagm(ones(2))))

# Newton trust region: exact AD Hessian with the Steihaug–Toint method
alg = QuasiNewtonAlg(NewtonTRApprox(); linesearch = :trustregion)
```

The `linesearch` keyword is `:backtracking` (Armijo, default), `:wolfe`
(strong Wolfe) or `:trustregion` (Steihaug–Toint with a projected step,
needed for `NewtonTRApprox`); `max_iter`, `tol`, `damping` (Powell damping)
and `Δ0` (initial trust-region radius) tune the driver.

### Preconditioners

`LBFGSApprox(memory; H0 = ...)` accepts a base operator used at the bottom
of the two-loop recursion:

```julia
alg = QuasiNewtonAlg(LBFGSApprox(10))                          # identity
alg = QuasiNewtonAlg(LBFGSApprox(10; H0 = DiagonalPreconditioner(d)))
```

[`IdentityPreconditioner`](@ref) applies the identity scaling;
[`DiagonalPreconditioner`](@ref)`(d)` multiplies by `d` (with entries all
positive) and divides when solving. See [Theory](theory.md) for how the
preconditioner enters the recursion.

### Custom Hessian approximations

Any subtype of [`AbstractHessianApprox`](@ref) plugs into
`QuasiNewtonAlg`. Implement the interface

```julia
struct MyApprox <: AbstractHessianApprox; ...; end
NonconvexAugLagLab.init!(H::MyApprox, n, x0)   # allocate/initialize
NonconvexAugLagLab.apply!(Hv, H::MyApprox, v)  # matrix-vector product
NonconvexAugLagLab.update!(H::MyApprox, s, y)  # accumulate the secant pair
NonconvexAugLagLab.solve!(d, H::MyApprox, g)   # solve the Newton system
Base.reset!(H::MyApprox)                       # drop accumulated curvature
```

Default implementations are provided for dense matrices, sparse matrices
(pattern-based) and diagonal scaling, so custom containers only need the
methods relevant to their structure. The full extension API is documented
in the docstrings of [`AbstractHessianApprox`](@ref).
