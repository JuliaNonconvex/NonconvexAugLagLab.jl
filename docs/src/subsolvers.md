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
(strong Wolfe), `:trustregion` (Steihaug–Toint with a projected step,
needed for `NewtonTRApprox`) or `:tron` (Lin–Moré projected CG with
active-set identification — see [Theory](theory.md)); `max_iter`, `tol`,
`damping` (Powell damping) and `Δ0` (initial trust-region radius) tune the
driver.

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
