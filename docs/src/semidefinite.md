## Semidefinite constraints

With `NonconvexSemidefinite` loaded, `AugLagAlg` additionally supports
semidefinite constraints through a projected augmented Lagrangian: the
constraint violation is computed through the PSD projection
(`project_psd`) and the multiplier update uses the projected multiplier
update of the PSD cone. A problem with a matrix-valued constraint is
modeled with `add_sd_constraint!`:

```julia
using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra

# min ‖X - A‖²_F s.t. X ⪰ 0 over the lower-triangular entries of X
A = [2.0 0.5; 0.5 1.0]
m = Model((x_l, x_d) -> sum(abs2, decompress_symmetric(x_l, x_d) .- A))
addvar!(m, fill(-5.0, 1), fill(5.0, 1), fill(0.0, 2)) # 1 lower + 2 diagonal entries
add_sd_constraint!(m, (x_l, x_d) -> decompress_symmetric(x_l, x_d))
r = optimize(m, AugLagAlg(), [fill(1.0, 1), fill(1.0, 2)])
X = decompress_symmetric(r.minimizer...)
X  # ≈ the PSD projection of A
```

The semidefinite relaxation is controlled like the other constraints
through [`RelaxConstraintSelection`](@ref): `sd = false` keeps the
semidefinite constraints hard for the sub-solver, `sd = true` (default
when relaxing) folds them into the augmented Lagrangian. The multiplier
update for the `:pbm` and `:mixed` penalty modes also barrier-ifies
inequality constraints; the semidefinite terms are always handled by the
projected update. See [Theory](theory.md) for the derivation.

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
