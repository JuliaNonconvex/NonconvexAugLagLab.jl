# NonconvexAugLagLab



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

See [Quick start](quickstart.md) for the first steps, [Options](options.md) for
the penalty modes and the relaxation selection, [Semidefinite
constraints](semidefinite.md) for PSD-constrained problems,
[Continuation](continuation.md) for warm-continued solves,
[Quasi-Newton sub-solver](subsolvers.md) for the built-in sub-solver,
[Result](result.md) for the fields of the returned result, [Theory](theory.md)
for the algorithmic background and [Docstrings](docstrings.md) for the full
API reference.
