using NonconvexAugLagLab, NonconvexIpopt, Test, LinearAlgebra

@testitem "AugLagAlg: equality constraint" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # min x₁² + x₂² s.t. x₁ + x₂ = 1 → (0.5, 0.5), value 0.5
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r = optimize(m, AugLagAlg(), [0.0, 0.0])
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test abs(r.minimum - 0.5) < 1e-4
end

@testitem "AugLagAlg: inequality constraint" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # min x₁² + x₂² s.t. 2 - x₁ - x₂ ≤ 0 → (1, 1), value 2
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
    r = optimize(m, AugLagAlg(), [1.5, 1.5])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-4
    @test abs(r.minimum - 2.0) < 1e-4
end

@testitem "AugLagAlg: mixed equality + inequality" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # min x₁² + x₂² s.t. x₁ + x₂ = 1 and x₁ ≥ 0.75 → (0.75, 0.25)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> 0.75 - x[1])
    r = optimize(m, AugLagAlg(), [0.5, 0.5])
    @test norm(r.minimizer - [0.75, 0.25]) < 1e-4
end

@testitem "AugLagAlg: PBM penalty mode" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
    options = AugLagOptions(penalty=:pbm)
    r = optimize(m, AugLagAlg(), [1.5, 1.5], options=options)
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-4
end

@testitem "AugLagAlg: keep constraints hard for the sub-solver" begin
    using NonconvexAugLagLab, NonconvexIpopt, Test, LinearAlgebra
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

@testitem "AugLagAlg: the built-in solver requires full relaxation" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    options = AugLagOptions(relaxed=RelaxConstraintSelection(eq=false))
    @test_throws ArgumentError optimize(m, AugLagAlg(), [0.0, 0.0], options=options)
end

@testitem "AugLagAlg: keep_all trajectory" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    options = AugLagOptions(keep_all=true)
    r = optimize(m, AugLagAlg(), [0.0, 0.0], options=options)
    @test r.trajectory !== nothing
    @test length(r.trajectory) >= 1
end
