<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexAugLagLab.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexAugLagLab.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexAugLagLab.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexAugLagLab.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexAugLagLab.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexAugLagLab.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexAugLagLab.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexAugLagLab.jl

Augmented Lagrangian laboratory

Classical ALM and penalty-barrier-multiplier methods with pluggable quasi-Newton sub-solvers.

NonconvexAugLagLab.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl) under the [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl) umbrella.

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

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexAugLagLab.jl/stable](https://julianonconvex.github.io/NonconvexAugLagLab.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexAugLagLab.jl/dev](https://julianonconvex.github.io/NonconvexAugLagLab.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexAugLagLab")
```
