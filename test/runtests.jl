using NonconvexAugLagLab
using Test, LinearAlgebra

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a*x[1] + b)^3 - x[2]

m = Model(f)
addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
add_ineq_constraint!(
    m,
    FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2),
)
x0 = [2.0, 2.0]

#alg = AugLag2(primaloptimizer = MMA87())
alg = AugLag2()
options = AugLag2Options(alg)
r = optimize(m, alg, x0, options = options)
@test abs(r.minimum - sqrt(8/27)) < 1e-4
@test norm(r.minimizer - [1/3, 8/27]) < 1e-4
