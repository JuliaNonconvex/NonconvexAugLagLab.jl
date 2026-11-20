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
