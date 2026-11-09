using NonconvexAugLagLab
using Test, LinearAlgebra

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

m = Model(f)
addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))
x0 = [2.0, 2.0]

#alg = AugLag2(primaloptimizer = MMA87())
alg = AugLag2()
options = AugLag2Options(alg)
r = optimize(m, alg, x0; options=options)
@test abs(r.minimum - sqrt(8 / 27)) < 1e-4
@test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-4

@testset "Equality constraints are rejected" begin
    # Regression test: equality constraints used to be silently ignored.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    @test_throws ArgumentError optimize(m, AugLag2(), [0.0, 0.0])
end

@testset "Unconstrained models do not crash" begin
    # Regression test: the residuals used to call maximum on an empty g.
    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, AugLag2(), [0.0, 0.0], options=AugLag2Options(AugLag2(); maxiter=3))
    @test norm(r.minimizer - [1.0, -2.0]) < 1e-3
end

using Aqua, JET
Aqua.test_all(NonconvexAugLagLab)
JET.test_package(NonconvexAugLagLab; target_modules = (NonconvexAugLagLab,))
