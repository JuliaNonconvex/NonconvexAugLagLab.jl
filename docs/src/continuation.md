## Continuation (`warm_continue`)

A SIMP-style continuation — solve, change a model parameter, solve again —
can reuse the whole optimization state through one workspace:

```julia
using NonconvexAugLagLab, NonconvexCore
vm, x0, _ = NonconvexCore.tovecmodel(m, init)
options = AugLagOptions(max_iter = 100, warm_continue = true)
ws = Workspace(vm, AugLagAlg(), x0; options = options)

r1 = NonconvexCore.optimize!(ws)     # stage 1
setpenalty!(penalty, p)              # e.g. TopOpt: change the SIMP exponent
r2 = NonconvexCore.optimize!(ws)     # continues from r1's state and curvature
```

The second call restarts from the previous final iterate and reuses the
multipliers (rescaled to the new constraint scales), the per-group penalty
weights and the L-BFGS curvature; `ρ`, `η`, `ω`, the filter and the Anderson
history re-initialize from the new options, and a structure mismatch falls
back to a cold start. This pays off when the parameter change between
solves is small — measured on the TopOpt local-stress SIMP continuation, a
fine 6-step schedule ends both faster and better than cold restarts.
