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
