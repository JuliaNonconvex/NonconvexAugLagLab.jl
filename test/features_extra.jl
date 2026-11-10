@testitem "AugLagOptions: penalty=:mixed and penalty-mode validation" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # min x₁² + x₂² s.t. x₁ + x₂ = 1 and 0.75 - x₁ ≤ 0 → (0.75, 0.25).
    mixed() = begin
        m = Model(x -> sum(x .^ 2))
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        add_eq_constraint!(m, x -> x[1] + x[2] - 1)
        add_ineq_constraint!(m, x -> 0.75 - x[1])
        return m
    end
    # min x₁² + x₂² s.t. 2 - x₁ - x₂ ≤ 0 → (1, 1).
    ineq_only() = begin
        m = Model(x -> sum(x .^ 2))
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
        return m
    end
    # quadratic: the classical ALM converges to the exact optimum
    rq = optimize(
        mixed(), AugLagAlg(), [0.5, 0.5]; options=AugLagOptions(penalty=:quadratic)
    )
    @test norm(rq.minimizer - [0.75, 0.25]) < 1e-4
    @test rq isa AugLagResult
    # pbm and mixed on an inequality-only problem converge to (1, 1)
    for penalty in (:pbm, :mixed)
        ri = optimize(
            ineq_only(), AugLagAlg(), [1.5, 1.5]; options=AugLagOptions(penalty=penalty)
        )
        @test norm(ri.minimizer - [1.0, 1.0]) < 1e-4
        @test ri isa AugLagResult
    end
    # pbm and mixed on the mixed problem: the finite-penalty PBM barrier
    # biases the stationary point slightly into the feasible side; assert
    # feasibility and closeness of the objective instead of the exact point
    for penalty in (:pbm, :mixed)
        rm = optimize(
            mixed(), AugLagAlg(), [0.5, 0.5]; options=AugLagOptions(penalty=penalty)
        )
        @test abs(rm.minimizer[1] + rm.minimizer[2] - 1) < 1e-3
        @test 0.75 - rm.minimizer[1] < 1e-3
        @test rm.minimum < 0.7
    end
    @test_throws ArgumentError AugLagOptions(penalty=:foo)
    @test_throws ArgumentError AugLagOptions(ρ_init=0.0)
    @test_throws ArgumentError AugLagOptions(ρ_incr=1.0)
    @test_throws ArgumentError AugLagOptions(ρ_decr=0.5)
end

@testitem "AugLagOptions: weights, lsq_multipliers and custom penalty schedule" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # weights needs one entry per relaxed constraint *group*: one equality
    # and one inequality are relaxed here. Compatible pair: optimum (0.75, 0.25).
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> 0.75 - x[1])
    options = AugLagOptions(
        weights=[2.0],
        lsq_multipliers=true,
        ρ_init=1.0,
        ρ_max=1e6,
        ρ_incr=2.0,
        ρ_decr=4.0,
        max_iter=100,
        ctol=1e-6,
        η₀=0.9,
        ω₀=1.5,
        keep_all=true,
    )
    r = optimize(m, AugLagAlg(), [0.5, 0.5], options=options)
    @test norm(r.minimizer - [0.75, 0.25]) < 1e-4
    @test r.trajectory !== nothing
    @test all(t -> length(t[1]) == 2, r.trajectory)
    @test_throws ArgumentError optimize(
        m, AugLagAlg(), [0.5, 0.5]; options=AugLagOptions(weights=[1.0, 2.0])
    )
end

@testitem "RelaxConstraintSelection: index vectors" begin
    using NonconvexAugLagLab, NonconvexIpopt, Test, LinearAlgebra
    # relax only the first inequality constraint; keep the second hard for
    # Ipopt (it supports inequalities natively). The hard constraint
    # 2 - x₂ ≤ 0 forces x₂ ≥ 2, so the optimum is (0, 2) with value 4. The
    # start is feasible for both constraints so the sub-solves stay in the
    # regular (non-restoration) regime.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> 2 - x[1] - x[2])   # relaxed
    add_ineq_constraint!(m, x -> 2 - x[2])          # kept hard: x₂ ≥ 2
    sel = RelaxConstraintSelection(eq=false, ineq=[1], sd=false)
    @test sel.eq === false
    @test sel.ineq == [1]
    alg = AugLagAlg(IpoptAlg())
    options = AugLagOptions(
        relaxed=sel, sub_options=IpoptOptions(max_iter=500, print_level=0)
    )
    r = optimize(m, alg, [1.0, 2.5], options=options)
    @test norm(r.minimizer - [0.0, 2.0]) < 1e-3
end

@testitem "QuasiNewtonAlg: Wolfe linesearch, damping and driver options" begin
    using NonconvexAugLagLab, Test, LinearAlgebra, SparseArrays
    rosen = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    for linesearch in (:backtracking, :wolfe)
        m = Model(rosen)
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        alg = QuasiNewtonAlg(; linesearch=linesearch, max_iter=500, tol=1e-8)
        r = optimize(m, alg, [-1.2, 1.0])
        @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    end
    # damping off + explicit identity preconditioner + trust-region Δ0
    m = Model(rosen)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    alg = QuasiNewtonAlg(LBFGSApprox(5; H0=IdentityPreconditioner()); damping=false)
    r = optimize(m, alg, [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    @test_throws ArgumentError QuasiNewtonAlg(; linesearch=:foo)
    # each approximation type is the right AbstractHessianApprox subtype
    @test LBFGSApprox(3) isa AbstractHessianApprox
    @test DenseBFGSApprox(form=:inverse) isa AbstractHessianApprox
    @test DenseBFGSApprox(form=:hessian) isa AbstractHessianApprox
    @test StructuredBFGSApprox(spdiagm(ones(3))) isa AbstractHessianApprox
    @test NewtonTRApprox() isa AbstractHessianApprox
    @test IdentityPreconditioner() isa IdentityPreconditioner
    @test DiagonalPreconditioner(ones(3)) isa DiagonalPreconditioner
end

@testitem "AugLagAlg with an explicit QuasiNewtonAlg sub-solver" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    alg = AugLagAlg(QuasiNewtonAlg(LBFGSApprox(10)))
    r = optimize(m, alg, [0.0, 0.0])
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test abs(r.minimum - 0.5) < 1e-4
end

@testitem "AugLagResult fields and default trajectory" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r = optimize(m, AugLagAlg(), [0.0, 0.0])
    @test r isa AugLagResult
    @test length(r.minimizer) == 2
    @test isfinite(r.minimum)
    @test r.trajectory === nothing   # keep_all defaults to false
end
