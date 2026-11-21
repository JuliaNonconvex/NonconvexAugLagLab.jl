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

@testitem "AugLagAlg: hard constraints via the built-in penalty split" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # keeping the inequality hard folds it into the built-in subproblem as
    # a quadratic penalty (hard_rho); the problem still converges.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> [x[1] - 0.9], [0.0])
    r = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options = AugLagOptions(;
            max_iter = 200, relaxed = RelaxConstraintSelection(eq = true, ineq = false)
        ),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test r.minimum ≈ 0.5 atol = 1e-4
    # hard_rho must be positive
    @test_throws ArgumentError AugLagOptions(hard_rho = 0.0)
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
        options=AugLagOptions(; max_iter=20, rho_init=10.0, rho_decr=10.0),
    )
    @test abs(r.minimizer[1] - 1.0) < 1e-4
end

@testitem "AugLagAlg: Birgin-Martinez nonmonotone watchdog" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    x0 = [0.0, 0.0]
    # watchdog alone
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, watchdog=true))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # watchdog combined with the filter
    r = optimize(
        m,
        AugLagAlg(),
        x0;
        options=AugLagOptions(max_iter=40, watchdog=true, watchdog_tau=0.1, filter=true),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    # validation
    @test_throws ArgumentError AugLagOptions(watchdog_tau=0.0)
    @test_throws ArgumentError AugLagOptions(watchdog_tau=1.0)
end

@testitem "AugLagAlg: Birgin-Martinez outer tolerance schedule" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    x0 = [0.0, 0.0]
    r = optimize(m, AugLagAlg(), x0; options=AugLagOptions(max_iter=40, outer_schedule=:bm))
    @test norm(r.minimizer - [0.5, 0.5]) < 5e-2
    @test_throws ArgumentError AugLagOptions(outer_schedule=:foo)
end

@testitem "AugLagAlg: watchdog safety net rejects merit-increasing subproblem steps" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra

    # A deliberately lazy subsolver: it always returns a step that increases
    # the AL merit (moves away from the optimum *and* away from feasibility),
    # mimicking a subsolver that fails to make progress.
    struct AlwaysWorseAlg <: NonconvexCore.AbstractOptimizer end
    struct AlwaysWorseResult{X} <: NonconvexCore.AbstractResult
        minimizer::X
        minimum::Float64
    end
    NonconvexCore.solver_capabilities(::AlwaysWorseAlg) =
        ConstraintCapabilities(; box=true, continuous=true, infinite_bounds=true)
    struct AlwaysWorseWorkspace{M,X} <: Workspace
        model::M
        x0::X
    end
    function Workspace(model::NonconvexCore.VecModel, ::AlwaysWorseAlg, x0; kwargs...)
        return AlwaysWorseWorkspace(model, copy(x0))
    end
    function NonconvexCore.optimize!(ws::AlwaysWorseWorkspace)
        xnew = ws.x0 .- 0.1 .* ones(length(ws.x0))
        return AlwaysWorseResult(xnew, ws.model.objective(xnew))
    end

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    x0 = [0.0, 0.0]
    # Without the watchdog the worsening step is accepted every iteration and
    # the iterate drifts along the bad direction.
    r_off = optimize(
        m, AugLagAlg(AlwaysWorseAlg()), x0; options=AugLagOptions(max_iter=5, filter=false)
    )
    # With the watchdog, the first step gets the free pass (the reference
    # starts at +Inf), but every later worsening step is rejected outright and
    # the iterate stays put while the penalty escalates.
    r_on = optimize(
        m,
        AugLagAlg(AlwaysWorseAlg()),
        x0;
        options=AugLagOptions(max_iter=5, filter=false, watchdog=true),
    )
    @test norm(r_off.minimizer, Inf) > 1e-2
    @test norm(r_on.minimizer, Inf) <= 0.2
    @test norm(r_on.minimizer, Inf) < norm(r_off.minimizer, Inf) / 2
    # the slack parameter is validated
    @test_throws ArgumentError AugLagOptions(watchdog_tol=-0.1)
end

@testitem "AugLagAlg: newton exact-Hessian subproblem solves (both sources)" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra, JuMP
    # JuMP-converted model: the Hessian comes from the MOI evaluator
    jm = JuMP.Model()
    JuMP.@variable(jm, -5 <= x[1:2] <= 5, start=0.0)
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 1)^2)
    JuMP.@constraint(jm, x[1]^2 + x[2] - 2 == 0)   # nonlinear → expression path
    JuMP.@constraint(jm, x[1]^2 - x[2] <= 1.0)      # nonlinear ineq (evaluator)
    dm = DictModel(jm)
    x0 = NonconvexCore.getinit(dm)
    r = optimize(
        dm, AugLagAlg(), x0; options=AugLagOptions(max_iter=100, curvature=:newton)
    )
    @test norm(r.minimizer[:x] - [1.0, 1.0]) < 1e-2
    # plain model: the Hessian comes from AD
    m = NonconvexCore.Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    r2 = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=40, curvature=:newton)
    )
    @test norm(r2.minimizer - [0.5, 0.5]) < 5e-2
    # guard rails: PBM penalties are not supported
    @test_throws ArgumentError optimize(
        dm, AugLagAlg(), x0; options=AugLagOptions(curvature=:newton, penalty=:pbm)
    )
end

@testitem "AugLagAlg: box barrier subproblem mode" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=40, box_barrier=true)
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
    @test_throws ArgumentError AugLagOptions(box_barrier_scale=0.0)
end

@testitem "AugLagAlg: auto rho_init calibration" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    r = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=40, auto_rho_init=true)
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
end

@testitem "AugLagAlg: adaptive PBM barrier coefficient" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 0.5)
    r = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=40, penalty=:pbm, pbm_adaptive=true),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
end

@testitem "AugLagAlg: restoration phase recovers from a blow-up" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra
    # A subsolver that always returns NaNs: without the restoration phase
    # the AL reverts to the initial point; with it, a feasibility-only solve
    # recovers a near-feasible iterate.
    struct NaNAlg <: NonconvexCore.AbstractOptimizer end
    struct NR{X} <: NonconvexCore.AbstractResult
        minimizer::X
        minimum::Float64
    end
    NonconvexCore.solver_capabilities(::NaNAlg) =
        ConstraintCapabilities(; box=true, continuous=true, infinite_bounds=true)
    struct NW{M,X} <: Workspace
        model::M
        x0::X
    end
    function Workspace(model::NonconvexCore.VecModel, ::NaNAlg, x0; kwargs...)
        return NW(model, copy(x0))
    end
    function NonconvexCore.optimize!(ws::NW)
        return NR(fill(NaN, length(ws.x0)), NaN)
    end
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r_off = optimize(
        m, AugLagAlg(NaNAlg()), [0.0, 0.0]; options=AugLagOptions(max_iter=3, filter=false)
    )
    r_on = optimize(
        m,
        AugLagAlg(NaNAlg()),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=3, filter=false, restore=true),
    )
    @test abs(r_off.minimizer[1] + r_off.minimizer[2] - 1) > 0.5   # stayed at x0
    @test abs(r_on.minimizer[1] + r_on.minimizer[2] - 1) < 1e-6    # restored feasible
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
            max_iter=20,
            rho_init=10.0,
            rho_incr=1.2,
            rho_max=12.0,
            rho_decr=10.0,
            omega0=0.1,
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
            rho_init=10.0,
            scaling=false,
            sub_options=(; max_iter=300, tol=1e-8, globalization=:trustregion),
        ),
    )
    vm = @view r.minimizer[1:3]
    @test maximum(abs.(vm .- 1.0)) < 0.12
    @test abs(acopf_cons(r.minimizer)[2]) < 5e-2  # P balance at bus 2
end

@testitem "AugLagAlg: penalty-update helpers (escalation and two-way decrease)" begin
    using NonconvexAugLagLab, NonconvexCore, Test
    const NC = NonconvexCore
    opts = AugLagOptions(; rho_init=10.0, rho_incr=10.0, rho_max=1e4, eta0=0.5, ctol=1e-6)
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
    # the default schedule is :bm (full 1/ρ reset)
    @test η2 == max(opts.eta0 / ρ2, opts.ctol)
    @test ω2 == max(1 / ρ2, sqrt(eps(Float64)))
    # the gentler :default schedule is still available
    opts_def = AugLagOptions(;
        rho_init=10.0,
        rho_incr=10.0,
        rho_max=1e4,
        eta0=0.5,
        ctol=1e-6,
        outer_schedule=:default,
    )
    ρ3, η3, ω3 = NonconvexAugLagLab._escalate_penalty!(
        10.0,
        0.5,
        1.0,
        [1000.0],
        Float64[],
        1000.0,
        opts_def,
        [1],
        Int[],
        eq_fs,
        ineq_fs,
        [5.0],
    )
    @test η3 == max(opts_def.eta0 / ρ3^0.1, opts_def.ctol)
    @test ω3 == max(1 / ρ3^0.1, sqrt(eps(Float64)))
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

@testitem "AugLagAlg: stall-triggered restoration" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra

    # A sub-solver that returns its start point unchanged: the outer
    # violation never improves, so the stall trigger fires and the
    # built-in restoration must solve the feasibility-only problem.
    struct StuckAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::StuckAlg) =
        ConstraintCapabilities(; box=true, continuous=true, infinite_bounds=true)
    struct StuckResult{X} <: NonconvexCore.AbstractResult
        minimizer::X
        minimum::Float64
    end
    struct StuckWS{M,X} <: Workspace
        model::M
        x0::X
    end
    function Workspace(model::NonconvexCore.VecModel, ::StuckAlg, x0; kwargs...)
        return StuckWS(model, copy(x0))
    end
    NonconvexCore.optimize!(ws::StuckWS) = StuckResult(copy(ws.x0), 0.0)

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    # restoration disabled: the stuck solver leaves the iterate infeasible
    r_off = optimize(
        m,
        AugLagAlg(StuckAlg()),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=10, filter=false, restore=true, restore_stall=0),
    )
    @test abs(sum(r_off.minimizer) - 1) > 0.5
    # stall-triggered restoration (a very lax tolerance guarantees firing):
    # the reported point is the feasibility-restored one
    r_on = optimize(
        m,
        AugLagAlg(StuckAlg()),
        [0.0, 0.0];
        options=AugLagOptions(
            max_iter=10, filter=false, restore=true, restore_stall=2, restore_tol=0.9
        ),
    )
    @test abs(sum(r_on.minimizer) - 1) < 1e-4
    # the anchor keeps the restored point near the pre-restoration iterate
    r_anch = optimize(
        m,
        AugLagAlg(StuckAlg()),
        [0.0, 0.0];
        options=AugLagOptions(
            max_iter=10,
            filter=false,
            restore=true,
            restore_stall=2,
            restore_tol=0.9,
            restore_prox=1e2,
        ),
    )
    @test abs(sum(r_anch.minimizer) - 1) > abs(sum(r_on.minimizer) - 1)

    # option validation
    @test_throws ArgumentError AugLagOptions(restore_stall=-1)
    @test_throws ArgumentError AugLagOptions(restore_tol=0.0)
    @test_throws ArgumentError AugLagOptions(restore_tol_mode=:bad)
    @test_throws ArgumentError AugLagOptions(restore_tol=2.0, restore_tol_mode=:relative)
    @test_throws ArgumentError AugLagOptions(restore_prox=-1.0)
end

@testitem "AugLagAlg: raw violation metric" begin
    using NonconvexAugLagLab, NonconvexCore, Test

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - 2.0)
    vm, x0, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    # raw violation = max over the relaxed groups of the raw inf-norms:
    # eq |x1+x2-1| = 1, ineq (x1-2)^+ = 0 at x0
    @test NonconvexAugLagLab._raw_violation(
        vm, NonconvexAugLagLab.RelaxConstraintSelection(), x0
    ) ≈ 1.0
    # feasible point: both groups zero
    @test NonconvexAugLagLab._raw_violation(
        vm, NonconvexAugLagLab.RelaxConstraintSelection(), [1.0, 0.0]
    ) ≈ 0.0
    # infeasible inequality dominates: at (2.5, -1.5) the equality holds and
    # the raw violation is the inequality excess 0.5
    @test NonconvexAugLagLab._raw_violation(
        vm, NonconvexAugLagLab.RelaxConstraintSelection(), [2.5, -1.5]
    ) ≈ 0.5
end

@testitem "AugLagAlg: option branches (sr1, prox, watchdog, schedules, barrier guards)" begin
    using NonconvexAugLagLab, NonconvexCore, Test, LinearAlgebra

    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)

    # curvature = :sr1 (built-in sub-solver, default h0)
    r_sr1 = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=60, curvature=:sr1)
    )
    @test norm(r_sr1.minimizer - [0.5, 0.5]) < 1e-3

    # proximal regularization with the built-in sub-solver
    r_prox = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=60, prox=1.0)
    )
    @test norm(r_prox.minimizer - [0.5, 0.5]) < 1e-3

    # nonmonotone watchdog acceptance
    r_wd = optimize(
        m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=40, watchdog=true)
    )
    @test norm(r_wd.minimizer - [0.5, 0.5]) < 1e-3

    # the gentler :default outer schedule
    r_def = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=40, outer_schedule=:default),
    )
    @test norm(r_def.minimizer - [0.5, 0.5]) < 1e-3

    # the fixed subproblem tolerance schedule
    r_fix = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=40, sub_tol_schedule=:fixed),
    )
    @test norm(r_fix.minimizer - [0.5, 0.5]) < 1e-3

    # guard rails
    @test_throws ArgumentError optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options=AugLagOptions(max_iter=5, box_barrier=true, elastic=true),
    )
end

@testitem "AugLagAlg: restoration merit covers the inequality channel" begin
    using NonconvexAugLagLab, NonconvexCore, Test

    # The restoration's raw merit and its FD column scaling take an ineq
    # branch; a stall on an eq+ineq model exercises it.
    struct StuckAlg2 <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::StuckAlg2) =
        ConstraintCapabilities(; box=true, continuous=true, infinite_bounds=true)
    struct StuckResult2{X} <: NonconvexCore.AbstractResult
        minimizer::X
        minimum::Float64
    end
    struct StuckWS2{M,X} <: Workspace
        model::M
        x0::X
    end
    function Workspace(model::NonconvexCore.VecModel, ::StuckAlg2, x0; kwargs...)
        return StuckWS2(model, copy(x0))
    end
    NonconvexCore.optimize!(ws::StuckWS2) = StuckResult2(copy(ws.x0), 0.0)

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[1] - x[2] - 1)
    r = optimize(
        m,
        AugLagAlg(StuckAlg2()),
        [0.0, 0.0];
        options=AugLagOptions(
            max_iter=10, filter=false, restore=true, restore_stall=2, restore_tol=0.9
        ),
    )
    # the restoration solved min ||x1+x2-1||^2 + ||(x1-x2-1)^+||^2
    @test abs(sum(r.minimizer) - 1) < 1e-4
end

@testitem "AugLagAlg: box-barrier mu adaptation" begin
    using NonconvexAugLagLab, Test

    μ = 1.0
    l = [0.0, 0.0]
    u = [1.0, 1.0]
    δ = 1e-4
    # far from the bounds: no complementarity, mu decays
    μ_far = NonconvexAugLagLab._box_barrier_μ(μ, [0.5, 0.5], l, u, δ, 1.0)
    @test μ_far == max(μ / 2, 1e-4)
    # near a lower bound with strong implied duals: mu shrinks
    # (well-complemented, the barrier has done its job)
    μ_near = NonconvexAugLagLab._box_barrier_μ(μ, [2δ, 0.5], l, u, δ, 1.0)
    @test μ_near == 0.5
    # near an upper bound: same on the upper side
    μ_up = NonconvexAugLagLab._box_barrier_μ(μ, [0.5, 1.0 - 2δ], l, u, δ, 1.0)
    @test μ_up == 0.5
    # weak duals near a bound grow mu back up (to the floor at least)
    μ_grow = NonconvexAugLagLab._box_barrier_μ(1e-6, [2δ, 0.5], l, u, δ, 1.0)
    @test μ_grow > 1e-6
end

@testitem "AugLagAlg: Anderson mixing on inequality multipliers" begin
    using NonconvexAugLagLab, Test
    # min ‖x − [1,1]‖² s.t. x₁ ≤ 0.5, x₂ ≤ 0.5, x₁ + x₂ ≤ 1.5: the
    # unconstrained optimum is infeasible, two constraints are active at
    # [0.5, 0.5] and one is inactive — the mix runs on active+inactive rows.
    m = Model(x -> sum((x .- [1.0, 1.0]) .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> [x[1] - 0.5, x[2] - 0.5, x[1] + x[2] - 1.5], [0.0, 0.0, 0.0])
    r = optimize(
        m,
        AugLagAlg(),
        [1.0, 1.0];
        options = AugLagOptions(; max_iter = 200, anderson_ineq = true),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
    @test r.minimum ≈ 0.5 atol = 1e-4
    # depth option: deeper mixing converges, invalid depths are rejected
    r5 = optimize(
        m,
        AugLagAlg(),
        [1.0, 1.0];
        options = AugLagOptions(; max_iter = 200, anderson_ineq = true, anderson_depth = 5),
    )
    @test norm(r5.minimizer - [0.5, 0.5]) < 1e-3
    @test_throws ArgumentError AugLagOptions(anderson_depth = 0)
end

@testitem "AugLagAlg: relaxed multiplier step with loose subproblem solves" begin
    using NonconvexAugLagLab, Test
    # loose inner solves (tol = 1e-2) converge with the Eckstein-style
    # relaxed multiplier step, on eq and ineq multipliers alike.
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [0.0, 0.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> [x[1] - 1.5, x[2] - 2.5], [0.0, 0.0])
    r = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options = AugLagOptions(;
            max_iter = 200,
            multiplier_relax = true,
            sub_options = (; max_iter = 50, tol = 1e-2, globalization = :trustregion),
        ),
    )
    @test norm(r.minimizer - [1.0, 2.0]) < 1e-4
    @test r.minimum < 1e-6
end

@testitem "AugLagAlg: spectral penalty update" begin
    using NonconvexAugLagLab, Test
    # the spectral (Barzilai-Borwein) penalty update must keep convergence
    # on a mixed eq+ineq problem (the safeguard only fires on a credible
    # positive correlation, so the plain runs are the regression baseline).
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> [x[1] - 0.9], [0.0])
    r = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options = AugLagOptions(; max_iter = 200, spectral_rho = true),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test r.minimum ≈ 0.5 atol = 1e-4
end

@testitem "AugLagAlg: Broyden dual-secant correction" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    const NAL = NonconvexAugLagLab
    # the inverse secant model starts at -I (first step = plain ascent)
    b = Ref{Any}(nothing)
    NAL._dual_secant_init!(b, 2)
    @test b[] == -Matrix{Float64}(I, 2, 2)
    # one good-Broyden inverse update satisfies the secant condition C⁺Δc = Δy
    C = copy(b[])
    Δy = [0.3, -0.1]
    Δc = [0.1, 0.05]
    NAL._dual_secant_update!(C, Δy, Δc)
    @test C * Δc ≈ Δy atol = 1e-12
    # degenerate denominators leave C unchanged
    C2 = copy(b[])
    NAL._dual_secant_update!(C2, zeros(2), Δc)
    @test C2 == b[]
    # end-to-end: an equality-constrained solve converges with the correction
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    r = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options = AugLagOptions(; max_iter = 200, dual_secant = true),
    )
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-4
    @test r.minimum ≈ 0.5 atol = 1e-4
end

@testitem "AugLagAlg: max_time budget" begin
    using NonconvexAugLagLab, Test
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    # an exhausted budget stops before the first subproblem: the result is x0
    r = optimize(m, AugLagAlg(), [0.0, 0.0]; options = AugLagOptions(; max_time = 1e-9))
    @test r.minimizer == [0.0, 0.0]
    # validation
    @test_throws ArgumentError AugLagOptions(max_time = 0.0)
    # a generous budget converges normally
    r2 = optimize(
        m,
        AugLagAlg(),
        [0.0, 0.0];
        options = AugLagOptions(; max_time = Inf, max_iter = 200),
    )
    @test norm(r2.minimizer - [0.5, 0.5]) < 1e-4
end
