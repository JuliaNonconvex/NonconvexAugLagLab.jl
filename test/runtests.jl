using NonconvexAugLagLab, NonconvexIpopt
using Test, LinearAlgebra, SparseArrays

# ------------------------------------------------------------------
# QuasiNewtonAlg standalone
# ------------------------------------------------------------------
@testset "QuasiNewtonAlg: L-BFGS on the Rosenbrock function" begin
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(), [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    @test abs(r.minimum) < 1e-6
end

@testset "QuasiNewtonAlg: L-BFGS with a diagonal preconditioner" begin
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    alg = QuasiNewtonAlg(LBFGSApprox(10; H0=DiagonalPreconditioner([1.0, 100.0])))
    r = optimize(m, alg, [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
end

@testset "QuasiNewtonAlg: dense BFGS inverse (default) and Hessian form" begin
    for form in (:inverse, :hessian)
        m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        r = optimize(m, QuasiNewtonAlg(DenseBFGSApprox(form=form)), [-1.2, 1.0])
        @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    end
end

@testset "QuasiNewtonAlg: structured sparse BFGS" begin
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    # diagonal sparsity pattern: only the diagonal of the Hessian model is kept
    B = spdiagm(ones(2))
    r = optimize(m, QuasiNewtonAlg(StructuredBFGSApprox(B)), [-1.2, 1.0]; options=nothing)
    @test norm(r.minimizer - [1.0, 1.0]) < 5e-2
end

@testset "QuasiNewtonAlg: Newton trust region" begin
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(NewtonTRApprox(); linesearch=:trustregion), [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
end

# A custom Hessian approximation through the extension API: a fixed diagonal
# scaling that accumulates nothing.
struct FixedDiagonalApprox <: AbstractHessianApprox
    d::Vector{Float64}
end
function NonconvexAugLagLab.init!(H::FixedDiagonalApprox, n, x0)
    return H
end
function NonconvexAugLagLab.apply!(Hv, H::FixedDiagonalApprox, v)
    Hv .= H.d .* v
    return Hv
end
function NonconvexAugLagLab.update!(H::FixedDiagonalApprox, s, y)
    return H
end
function NonconvexAugLagLab.solve!(d, H::FixedDiagonalApprox, g)
    d .= g ./ H.d
    return d
end

@testset "QuasiNewtonAlg: user-defined Hessian approximation" begin
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(FixedDiagonalApprox([1.0, 1.0])), [3.0, 4.0])
    @test norm(r.minimizer) < 1e-3
end

# ------------------------------------------------------------------
# AugLagAlg
# ------------------------------------------------------------------
@testset "AugLagAlg: equality constraint" begin
    # min x₁² + x₂² s.t. x₁ + x₂ = 1 → (0.5, 0.5), value 0.5
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r = optimize(m, AugLagAlg(), [0.0, 0.0])
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test abs(r.minimum - 0.5) < 1e-4
end

@testset "AugLagAlg: inequality constraint" begin
    # min x₁² + x₂² s.t. 2 - x₁ - x₂ ≤ 0 → (1, 1), value 2
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
    r = optimize(m, AugLagAlg(), [1.5, 1.5])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-4
    @test abs(r.minimum - 2.0) < 1e-4
end

@testset "AugLagAlg: mixed equality + inequality" begin
    # min x₁² + x₂² s.t. x₁ + x₂ = 1 and x₁ ≥ 0.75 → (0.75, 0.25)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> 0.75 - x[1])
    r = optimize(m, AugLagAlg(), [0.5, 0.5])
    @test norm(r.minimizer - [0.75, 0.25]) < 1e-4
end

@testset "AugLagAlg: PBM penalty mode" begin
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
    options = AugLagOptions(penalty=:pbm)
    r = optimize(m, AugLagAlg(), [1.5, 1.5], options=options)
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-4
end

@testset "AugLagAlg: keep constraints hard for the sub-solver" begin
    # keep the equality hard (Ipopt handles it), relax nothing else
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    alg = AugLagAlg(IpoptAlg())
    options = AugLagOptions(
        relaxed=RelaxConstraintSelection(eq=false, ineq=true, sd=true),
        sub_options=IpoptOptions(max_iter=100),
    )
    r = optimize(m, alg, [0.0, 0.0], options=options)
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
end

@testset "AugLagAlg: the built-in solver requires full relaxation" begin
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    options = AugLagOptions(relaxed=RelaxConstraintSelection(eq=false))
    @test_throws ArgumentError optimize(m, AugLagAlg(), [0.0, 0.0], options=options)
end

@testset "AugLagAlg: keep_all trajectory" begin
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    options = AugLagOptions(keep_all=true)
    r = optimize(m, AugLagAlg(), [0.0, 0.0], options=options)
    @test r.trajectory !== nothing
    @test length(r.trajectory) >= 1
end

# ------------------------------------------------------------------
# Semidefinite relaxation (extension)
# ------------------------------------------------------------------
using NonconvexSemidefinite

@testset "AugLagAlg: semidefinite constraint (projected ALM)" begin
    # min (x - 3)² over the PSD cone of 1x1 matrices x ≥ 0 → x = 3... the
    # constraint is inactive; instead force x ≥ 2 via a 1x1 PSD constraint
    # x - 2 ≥ 0 and minimize (x - 3)² → x = 3 again (inactive). Use
    # min x² s.t. x ≥ 2 via PSD → x = 2.
    m = Model(x -> x[1]^2)
    addvar!(m, [-5.0], [5.0])
    add_sd_constraint!(m, x -> [x[1] - 2.0;;])
    r = optimize(m, AugLagAlg(), [0.0])
    @test abs(r.minimizer[1] - 2.0) < 1e-4
    @test abs(r.minimum - 4.0) < 1e-4
end

@testset "AugLagAlg: sd kept hard requires the sub-solver to support it" begin
    m = Model(x -> x[1]^2)
    addvar!(m, [-5.0], [5.0])
    add_sd_constraint!(m, x -> [x[1] - 2.0;;])
    options = AugLagOptions(
        relaxed=RelaxConstraintSelection(sd=false), sub_options=IpoptOptions(max_iter=10)
    )
    # Ipopt does not support sd constraints; the strict capability check errors.
    @test_throws ArgumentError optimize(m, AugLagAlg(IpoptAlg()), [0.0]; options=options)
end

@testset "solver capabilities" begin
    caps = solver_capabilities(AugLagAlg())
    @test caps.linear_eq === true
    @test caps.nonlinear_eq === true
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.linear_sd === true
    @test caps.nonlinear_sd === true
    @test caps.box === true
    @test caps.integer === false
    @test caps.infinite_bounds === true
end

include("aqua.jl")
