@testitem "adaptive subproblem tolerance: schedule and extension point" begin
    using Test, NonconvexCore, NonconvexAugLagLab
    using NonconvexIpopt: NonconvexIpopt

    const NAL = NonconvexAugLagLab
    # the fallback keeps the options unchanged
    @test NAL._set_sub_tol(nothing, (; max_iter=7), 1e-4) == (; max_iter=7)
    # the Ipopt extension (loaded when NonconvexIpopt is present) overrides
    # only the tolerance
    opts = NonconvexIpopt.IpoptOptions(max_iter=42, nlp_scaling_method="gradient-based")
    adapted = NAL._set_sub_tol(nothing, opts, 3e-5)
    @test adapted.nt.tol == 3e-5
    @test adapted.nt.max_iter == 42
    @test adapted.nt.nlp_scaling_method == "gradient-based"
    # schedule names are accepted (invalid ones are rejected by
    # _sub_tol_value, covered in the schedule-values testitem)
    @test AugLagOptions(sub_tol_schedule=:eta) isa AugLagOptions
    @test AugLagOptions(sub_tol_schedule=:eta2) isa AugLagOptions
    @test AugLagOptions(sub_tol_schedule=:fixed) isa AugLagOptions
    @test AugLagOptions(sub_tol_schedule=:relative) isa AugLagOptions

    # end-to-end with the Ipopt sub-solver and the default (:eta) schedule:
    # the sub-solver options get the adaptive tolerance each outer iteration
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> [x[1] - 1.5, x[2] - 2.5], [0.0, 0.0])
    r = optimize(
        m,
        AugLagAlg(NonconvexIpopt.IpoptAlg()),
        [0.0, 0.0];
        options=AugLagOptions(
            max_iter=50, sub_options=NonconvexIpopt.IpoptOptions(max_iter=200)
        ),
    )
    g = m.ineq_constraints.fs[1](r.minimizer)
    @test maximum(max.(g, 0.0)) < 1e-4
    @test r.minimum < 1.0 + 1e-4
end

@testitem "subproblem tolerance schedule values" begin
    using Test, NonconvexAugLagLab
    const NAL = NonconvexAugLagLab
    # :fixed
    @test NAL._sub_tol_value(:fixed, nothing, 0.5, 1.0, 0.1) == 1e-8
    # :eta2 clamps to [1e-10, 1e-8]
    @test NAL._sub_tol_value(:eta2, nothing, 0.5, 1.0, 0.1) == 1e-8
    @test NAL._sub_tol_value(:eta2, nothing, 1e-4, 1.0, 0.1) == 1e-8
    @test NAL._sub_tol_value(:eta2, nothing, 1e-6, 1.0, 0.1) == 1e-10
    # :eta clamps to [1e-6, 1e-2]
    @test NAL._sub_tol_value(:eta, nothing, 0.5, 1.0, 0.1) == 1e-2
    @test NAL._sub_tol_value(:eta, nothing, 0.05, 1.0, 0.1) ≈ 2.5e-3
    @test NAL._sub_tol_value(:eta, nothing, 1e-4, 1.0, 0.1) == 1e-6
    # :relative: the inner tolerance tracks the current violation
    @test NAL._sub_tol_value(:relative, nothing, 0.5, 1e-2, 0.1) == 1e-3
    @test NAL._sub_tol_value(:relative, nothing, 0.5, 1e-6, 0.1) == 1e-7
    @test NAL._sub_tol_value(:relative, nothing, 0.5, 1e-12, 0.1) == 1e-10 # vprev floor
    @test NAL._sub_tol_value(:relative, nothing, 0.5, 1e3, 0.1) == 1e-2    # clamp ceiling
    # an explicit sub_options.tol override wins over any schedule
    @test NAL._sub_tol_value(:eta, 1e-9, 0.5, 1.0, 0.1) == 1e-9
    @test NAL._sub_tol_value(:fixed, 1e-9, 0.5, 1.0, 0.1) == 1e-9
    @test NAL._sub_tol_value(:relative, 1e-9, 0.5, 1e-2, 0.1) == 1e-9
    # invalid schedules are rejected
    @test_throws ArgumentError NAL._sub_tol_value(:bogus, nothing, 0.5, 1.0, 0.1)
    # sub_tol_rel must lie in (0, 1)
    @test AugLagOptions(sub_tol_schedule=:relative, sub_tol_rel=0.5) isa AugLagOptions
    @test_throws ArgumentError AugLagOptions(sub_tol_schedule=:relative, sub_tol_rel=0.0)
    @test_throws ArgumentError AugLagOptions(sub_tol_schedule=:relative, sub_tol_rel=1.0)
end

@testitem "MMA and NLopt tolerance extensions" begin
    using Test, NonconvexCore, NonconvexAugLagLab
    using NonconvexMMA: NonconvexMMA
    using NonconvexNLopt: NonconvexNLopt
    const NAL = NonconvexAugLagLab

    mma = NonconvexMMA.MMAOptions(maxiter=12)
    a = NAL._set_sub_tol(nothing, mma, 4e-5)
    @test a.tol.fabs == 4e-5
    @test a.tol.kkt == 4e-5
    @test a.maxiter == 12
    @test a.outer_maxiter == mma.outer_maxiter

    nl = NonconvexNLopt.NLoptOptions(ftol_rel=0.5)
    b = NAL._set_sub_tol(nothing, nl, 2e-3)
    @test b.nt.ftol_abs == 2e-3
    @test b.nt.xtol_abs == 2e-3
    @test b.nt.ftol_rel == 0.5
end
