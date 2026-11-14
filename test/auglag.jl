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

@testitem "AugLagAlg: label normalization and selection copying" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # scalar labels (Symbol, String) and label collections all normalize
    @test RelaxConstraintSelection(labels=:relax).labels == Set([:relax])
    @test RelaxConstraintSelection(labels="relax").labels == Set([:relax])
    @test RelaxConstraintSelection(labels=("relax", :soft)).labels == Set([:relax, :soft])
    @test RelaxConstraintSelection().labels === nothing
    # UnitRange kind selections survive the labels path (push! on a range
    # would throw) and a unit-range selection solves end-to-end
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_eq_constraint!(m, x -> x[1] - 0.5; flags=[:relax])
    sel = RelaxConstraintSelection(eq=1:1, ineq=false, sd=false, labels=[:relax])
    out = NonconvexAugLagLab._relaxed_indices(sel.eq, m.eq_constraints, sel.labels)
    @test out == [1, 2]
    # the labels path must not mutate the user's index vector
    idx = [1]
    sel2 = RelaxConstraintSelection(eq=idx, ineq=false, sd=false, labels=[:relax])
    out2 = NonconvexAugLagLab._relaxed_indices(sel2.eq, m.eq_constraints, sel2.labels)
    @test out2 == [1, 2]
    @test idx == [1]
    # end-to-end: min sum(x.^2) s.t. x1+x2=1, x1=0.5 -> (0.5, 0.5)
    r = optimize(m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(relaxed=sel))
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
end

@testitem "AugLagAlg: Anderson mixing blends residuals" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # R₁ ⊥ R₂: the simplex-optimal blend is exactly θ = (0.5, 0.5), whose
    # combined residual ½R₁ + ½R₂ has norm √½ < ‖R₂‖ = 1. The old code
    # solved a zero right-hand side and returned the raw update unchanged.
    hist = [([0.0, 0.0], [1.0, 0.0]), ([1.0, 1.0], [0.0, 1.0])]
    y_new = [1.0, 2.0]   # == Y₂ + R₂: the unaccelerated update
    mixed = NonconvexAugLagLab._anderson_mix(hist, y_new)
    @test mixed != y_new
    @test mixed ≈ 0.5 .* ([0.0, 0.0] .+ [1.0, 0.0]) .+ 0.5 .* ([1.0, 1.0] .+ [0.0, 1.0])
    # the mixed residual norm is smaller than the raw one
    @test norm(0.5 .* hist[1][2] .+ 0.5 .* hist[2][2]) < norm(hist[2][2])
    # single-entry and empty histories fall back to the raw update
    @test NonconvexAugLagLab._anderson_mix([hist[1]], [1.0, 0.0]) == [1.0, 0.0]
    @test NonconvexAugLagLab._anderson_mix(
        Tuple{Vector{Float64},Vector{Float64}}[], [1.0, 0.0]
    ) == [1.0, 0.0]
    # identical residuals make the normal equations singular: the
    # unaccelerated update survives instead of a SingularException
    hist2 = [([0.0, 0.0], [1.0, 0.0]), ([1.0, 1.0], [1.0, 0.0])]
    @test NonconvexAugLagLab._anderson_mix(hist2, [2.0, 1.0]) == [2.0, 1.0]
end

@testitem "AugLagAlg: first-iteration failure falls back to the initial point" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    fallback = NonconvexAugLagLab._failed_iterate_fallback
    T = Tuple{Vector{Float64},Real,Real}
    # empty trajectory (first iteration failed): use the initial point
    @test fallback(T[], [1.0, 2.0]) == [1.0, 2.0]
    # nonempty trajectory: use the last accepted point
    @test fallback([([3.0, 4.0], 0.1, 0.2)], [1.0, 2.0]) == [3.0, 4.0]
    # the fallback is a copy, not an alias of the inputs
    x0 = [1.0, 2.0]
    out = fallback(T[], x0)
    out[1] = 99.0
    @test x0 == [1.0, 2.0]
    # end-to-end: a sub-solver whose subproblems always return non-finite
    # iterates used to crash the outer loop with a BoundsError on the very
    # first subproblem; the iterate is now discarded and the run finishes
    # (escalating the penalty) from the initial point.
    struct _AlwaysFailAlg <: NonconvexCore.AbstractOptimizer
        calls::Base.RefValue{Int}
    end
    NonconvexCore.solver_capabilities(::_AlwaysFailAlg) =
        ConstraintCapabilities(; box=true, continuous=true, infinite_bounds=true)
    struct _AlwaysFailWorkspace <: NonconvexCore.Workspace
        model::NonconvexCore.VecModel
        alg::_AlwaysFailAlg
        x0::Vector{Float64}
    end
    function NonconvexCore.Workspace(
        m::NonconvexCore.VecModel, a::_AlwaysFailAlg, x0; kwargs...
    )
        return _AlwaysFailWorkspace(m, a, copy(x0))
    end
    function NonconvexCore.optimize!(ws::_AlwaysFailWorkspace)
        ws.alg.calls[] += 1
        n = length(ws.x0)
        return NonconvexCore.GenericResult(
            ws.alg, copy(ws.x0), fill(NaN, n), NaN, 1, false, Tolerance(), nothing, 1
        )
    end
    failalg = _AlwaysFailAlg(Ref(0))
    mf = Model(x -> sum(x .^ 2))
    addvar!(mf, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(mf, x -> x[1] + x[2] - 1)
    rf = optimize(mf, AugLagAlg(failalg), [0.0, 0.0]; options=AugLagOptions(max_iter=3))
    @test failalg.calls[] >= 3
    @test all(isfinite, rf.minimizer)
    @test isfinite(rf.minimum)
end

@testitem "AugLagAlg: elastic mode with position-indexing constraint functions" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # The filter backtrack evaluates the constraint violation on the
    # slack-augmented vector; the functions index the variable vector by
    # position, so elastic mode used to throw a DimensionMismatch here.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    options = AugLagOptions(; max_iter=30, elastic=true)
    r = optimize(m, AugLagAlg(), [0.0, 0.0]; options=options)
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-2
end

@testitem "AugLagAlg: coverage of the option branches" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    x0 = [0.0, 0.0]
    # scaling=false
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, scaling=false))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # h0=:identity
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, h0=:identity))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # sub_tol_schedule=:eta2
    r = optimize(
        m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, sub_tol_schedule=:eta2)
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # selective=false (global escalation)
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, selective=false))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # penalty=:pbm and :mixed (PBM inequality term in the AL objective)
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, penalty=:pbm))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, penalty=:mixed))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
end

@testitem "AugLagAlg: two-way penalty decrease and global escalation" begin
    using NonconvexAugLagLab, Test
    # min -x s.t. x = 1 from x0 = 1: the first subproblem pushes x to 1 + 1/ρ
    # (the violation *worsens*), which forces the global-escalation branch,
    # then a good multiplier step triggers the two-way ρ decrease.
    m = Model(x -> -x[1])
    addvar!(m, [-10.0], [10.0])
    add_eq_constraint!(m, x -> x[1] - 1)
    r = optimize(
        m,
        AugLagAlg(),
        [1.0];
        options=AugLagOptions(; max_iter=20, ρ_init=10.0, ρ_decr=10.0),
    )
    @test abs(r.minimizer[1] - 1.0) < 1e-4
end

@testitem "AugLagAlg: sd relaxation without NonconvexSemidefinite throws" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_sd_constraint!(m, x -> [x[1] 0.0; 0.0 x[2]])
    @test_throws ArgumentError optimize(m, AugLagAlg(), [1.0, 1.0])
end

@testitem "AugLagAlg: two-way penalty decrease and global escalation" begin
    using NonconvexAugLagLab, Test
    # min -6x s.t. x = 1 from x0 = 1: the first subproblems overshoot
    # (violation worsens), the selective weight caps, rho escalates, and a
    # later good step with rho > rho_init triggers the two-way decrease.
    m = Model(x -> -6 * x[1])
    addvar!(m, [-10.0], [10.0])
    add_eq_constraint!(m, x -> x[1] - 1)
    r = optimize(
        m,
        AugLagAlg(),
        [1.0];
        options=AugLagOptions(;
            max_iter=20, ρ_init=10.0, ρ_incr=1.2, ρ_max=12.0, ρ_decr=10.0, ω₀=0.1
        ),
    )
    @test abs(r.minimizer[1] - 1.0) < 1e-4
end

@testitem "AugLagAlg: LS-multiplier CG runs multiple sweeps" begin
    using NonconvexAugLagLab, Test
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_eq_constraint!(m, x -> x[1] + 2 * x[2] - 1.5)
    r = optimize(m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(; max_iter=60))
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
end

@testitem "AugLagAlg: elastic PBM/mixed inequality terms" begin
    using NonconvexAugLagLab, Test
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - 0.5)
    for p in (:pbm, :mixed)
        r = optimize(
            m,
            AugLagAlg(),
            [2.0, 2.0];
            options=AugLagOptions(; max_iter=40, penalty=p, elastic=true),
        )
        @test abs(r.minimizer[1] + r.minimizer[2] - 1) < 1e-3
        @test r.minimizer[1] <= 0.5 + 1e-3
    end
end

@testitem "AugLagAlg: 3-bus AC-OPF (escalation + two-way decrease)" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    const NC = NonconvexCore
    # A minimal 3-bus AC-OPF: slack at bus 1, loads at 2 and 3, two lines.
    # The eq-heavy problem drives the selective bumps, the global rho
    # escalation and a later two-way decrease (the branches the toy
    # problems converge past too quickly).
    function acopf_obj(x)
        # x = [vm(3); va(3); pg; qg] with slack va=0
        vm = @view x[1:3]
        va = @view x[4:6]
        pg = x[7]
        qg = x[8]
        return 10 * pg^2 + 5 * pg
    end
    function acopf_cons(x)
        vm = x[1:3]
        va = x[4:6]
        pg = x[7]
        qg = x[8]
        Pd = [0.0, 0.6, 0.5]
        Qd = [0.0, 0.1, 0.1]
        r = [0.01, 0.01]
        xx = [0.1, 0.1]
        g = r ./ (r .^ 2 .+ xx .^ 2)
        b = -xx ./ (r .^ 2 .+ xx .^ 2)
        pg_b = [pg, 0.0, 0.0]
        qg_b = [qg, 0.0, 0.0]
        δ12 = va[1] - va[2]
        δ23 = va[2] - va[3]
        p12 = vm[1]^2 * g[1] - vm[1] * vm[2] * (g[1] * cos(δ12) + b[1] * sin(δ12))
        q12 = -vm[1]^2 * b[1] + vm[1] * vm[2] * (b[1] * cos(δ12) - g[1] * sin(δ12))
        p21 = vm[2]^2 * g[1] - vm[2] * vm[1] * (g[1] * cos(-δ12) + b[1] * sin(-δ12))
        q21 = -vm[2]^2 * b[1] + vm[2] * vm[1] * (b[1] * cos(-δ12) - g[1] * sin(-δ12))
        p23 = vm[2]^2 * g[2] - vm[2] * vm[3] * (g[2] * cos(δ23) + b[2] * sin(δ23))
        q23 = -vm[2]^2 * b[2] + vm[2] * vm[3] * (b[2] * cos(δ23) - g[2] * sin(δ23))
        p32 = vm[3]^2 * g[2] - vm[3] * vm[2] * (g[2] * cos(-δ23) + b[2] * sin(-δ23))
        q32 = -vm[3]^2 * b[2] + vm[3] * vm[2] * (b[2] * cos(-δ23) - g[2] * sin(-δ23))
        return [
            pg_b[1] - Pd[1] - p12,
            pg_b[2] - Pd[2] - p21 - p23,
            pg_b[3] - Pd[3] - p32,
            qg_b[1] - Qd[1] - q12,
            qg_b[2] - Qd[2] - q21 - q23,
            qg_b[3] - Qd[3] - q32,
        ]
    end
    m = Model(acopf_obj)
    addvar!(
        m,
        [0.9, 0.9, 0.9, -2.0, -2.0, 0.0, 0.0, -1.0],
        [1.1, 1.1, 1.1, 2.0, 2.0, 0.0, 3.0, 3.0],
    )
    add_eq_constraint!(m, acopf_cons; dim=6)
    x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.5, 0.3]
    r = optimize(
        m,
        AugLagAlg(),
        x0;
        options=AugLagOptions(;
            max_iter=120,
            ρ_init=10.0,
            scaling=false,
            sub_options=(; max_iter=300, tol=1e-8, linesearch=:trustregion),
        ),
    )
    vm = @view r.minimizer[1:3]
    @test maximum(abs.(vm .- 1.0)) < 0.12
    @test abs(acopf_cons(r.minimizer)[2]) < 5e-2  # P balance at bus 2
end

@testitem "AugLagAlg: penalty-update helpers (escalation and two-way decrease)" begin
    using NonconvexAugLagLab, NonconvexCore, Test
    const NC = NonconvexCore
    opts = AugLagOptions(; ρ_init=10.0, ρ_incr=10.0, ρ_max=1e4, η₀=0.5, ctol=1e-6)
    # two-way decrease
    @test NonconvexAugLagLab._two_way_step(100.0, opts, true, 0.1, 0.5) == 10.0
    @test NonconvexAugLagLab._two_way_step(2.0, opts, true, 0.1, 0.5) == 2.0
    @test NonconvexAugLagLab._two_way_step(100.0, opts, false, 0.1, 0.5) == 100.0
    @test NonconvexAugLagLab._two_way_step(100.0, opts, true, 1.0, 0.5) == 100.0
    eq_fs = [NC.EqConstraint(x -> x[1], 0.0)]
    ineq_fs = []
    # selective bump: the violating group is below the weight cap -> the
    # weight grows and rho stays.
    w_eq = [1.0]
    ρ, η, ω = NonconvexAugLagLab._escalate_penalty!(
        10.0, 0.5, 1.0, w_eq, Float64[], 1000.0, opts, [1], Int[], eq_fs, ineq_fs, [5.0]
    )
    @test w_eq[1] == 10.0
    @test ρ == 10.0
    # global escalation: the weight is at the cap -> rho grows and the
    # tolerances reset.
    ρ2, η2, ω2 = NonconvexAugLagLab._escalate_penalty!(
        10.0, 0.5, 1.0, [1000.0], Float64[], 1000.0, opts, [1], Int[], eq_fs, ineq_fs, [5.0]
    )
    @test ρ2 == 100.0
    @test η2 == max(opts.η₀ / ρ2^0.1, opts.ctol)
    @test ω2 == max(1 / ρ2^0.1, sqrt(eps(Float64)))
end

@testitem "AugLagAlg: elastic quadratic inequality term" begin
    using NonconvexAugLagLab, Test
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - 0.5)
    r = optimize(
        m, AugLagAlg(), [2.0, 2.0]; options=AugLagOptions(; max_iter=40, elastic=true)
    )
    @test abs(r.minimizer[1] + r.minimizer[2] - 1) < 1e-3
    @test r.minimizer[1] <= 0.5 + 1e-3
end
