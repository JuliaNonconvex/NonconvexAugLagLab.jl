# Augmented Lagrangian in pure Julia

## Description

`NonconvexAugLagLab.jl` provides `AugLag2`, an augmented Lagrangian
meta-algorithm for inequality-constrained problems. For a fixed quadratic
penalty weight it maximizes the augmented Lagrangian dual over the
non-negative multipliers and solves the primal subproblems with MMA or an
Optim-compatible algorithm. Equality constraints are not supported.

## Quick start

```julia
using NonconvexAugLagLab, NonconvexMMA

alg = AugLag2(primaloptimizer = MMA02(), dualoptimizer = Optim.GradientDescent())
options = AugLag2Options(alg)
result = optimize(model, alg, x0, options = options)
```

## Options

The options keyword argument to the `optimize` function must be an instance of
`AugLag2Options`:
- `primaloptions`: options of the primal sub-solver,
- `dualoptions`: options of the dual optimizer,
- `maxiter`: number of outer augmented Lagrangian iterations,
- `tol`: convergence tolerance,
- `quadfactor`: factor multiplying the quadratic penalty weight each iteration.
