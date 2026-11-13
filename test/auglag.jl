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

@testitem "AugLagAlg: warm_continue reuses state across solves" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    const NC = NonconvexCore
    # A two-stage chained solve with warm_continue=true: the second call
    # must reuse the curvature object and beat a cold restart of equal
    # total budget.
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [0.0, 0.0], [Inf, Inf])
    add_eq_constraint!(m, x -> x[1] + x[2] - 2)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.1)
    vm, _, _ = NC.tovecmodel(m, [0.5, 0.5])
    opts = AugLagOptions(; max_iter=60, warm_continue=true)
    ws = Workspace(vm, AugLagAlg(), [0.5, 0.5]; options=opts)
    r1 = NC.optimize!(ws)
    a1 = ws.state.approx
    @test ws.state !== nothing
    @test a1 !== nothing
    # second call: same workspace, warm state
    r2 = NC.optimize!(ws)
    @test ws.state.approx === a1  # curvature object carried over
    @test norm(r2.minimizer - [1.0, 1.0]) < 1e-2
    # a cold run of the same total budget must not do better
    rc = optimize(vm, AugLagAlg(), [0.5, 0.5]; options=AugLagOptions(; max_iter=120))
    @test r2.minimum <= rc.minimum + 1e-3
    # structure mismatch falls back to a cold start without error
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m2, x -> x[1] + x[2] - 1)
    vm2, _, _ = NC.tovecmodel(m2, [0.0, 0.0])
    ws2 = Workspace(
        vm2,
        AugLagAlg(),
        [0.0, 0.0];
        options=AugLagOptions(; max_iter=40, warm_continue=true),
    )
    r3 = NC.optimize!(ws2)
    @test norm(r3.minimizer - [0.5, 0.5]) < 1e-2
end

@testitem "AugLagAlg: relax by constraint labels" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # min sum(x.^2) s.t. x[1] + x[2] = 1 (labeled :relax) and x[1] <= 0.5
    # (not labeled). Relaxing only the labeled constraint must relax the
    # equality and keep the inequality hard -> the built-in solver then
    # refuses (it needs everything relaxed), which proves the selection
    # worked.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1; flags=[:relax])
    add_ineq_constraint!(m, x -> x[1] - 0.5)
    sel = RelaxConstraintSelection(eq=false, ineq=false, sd=false, labels=[:relax])
    options = AugLagOptions(relaxed=sel)
    @test_throws ArgumentError optimize(m, AugLagAlg(), [0.0, 0.0], options=options)
    # with both constraints labeled, everything is relaxed and the solve
    # succeeds: min sum(x.^2) s.t. x1+x2=1, x1<=0.5 -> (0.5, 0.5)
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m2, x -> x[1] + x[2] - 1; flags=[:relax])
    add_ineq_constraint!(m2, x -> x[1] - 0.5; flags=[:relax])
    r = optimize(m2, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(relaxed=sel))
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
    # a single Symbol works too, and kind+label selections combine
    m3 = Model(x -> sum(x .^ 2))
    addvar!(m3, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m3, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m3, x -> x[1] - 0.5; flags=[:relax])
    sel3 = RelaxConstraintSelection(eq=true, ineq=false, sd=false, labels=:relax)
    r3 = optimize(m3, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(relaxed=sel3))
    @test norm(r3.minimizer - [0.5, 0.5]) < 1e-3
end
