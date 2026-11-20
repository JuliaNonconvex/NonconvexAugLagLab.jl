@testitem "adaptive subproblem tolerance: schedule and extension point" begin
    using Test, NonconvexCore, NonconvexAugLagLab
    using NonconvexIpopt: NonconvexIpopt

    const NAL = NonconvexAugLagLab
    # the fallback keeps the options unchanged
    @test NAL._set_sub_tol(nothing, (; max_iter = 7), 1e-4) == (; max_iter = 7)
    # the Ipopt extension (loaded when NonconvexIpopt is present) overrides
    # only the tolerance
    opts = NonconvexIpopt.IpoptOptions(max_iter = 42, nlp_scaling_method = "gradient-based")
    adapted = NAL._set_sub_tol(nothing, opts, 3e-5)
    @test adapted.nt.tol == 3e-5
    @test adapted.nt.max_iter == 42
    @test adapted.nt.nlp_scaling_method == "gradient-based"
    # invalid schedule names are rejected
    @test_throws ArgumentError AugLagOptions(sub_tol_schedule = :bogus)
    @test AugLagOptions(sub_tol_schedule = :eta) isa AugLagOptions
    @test AugLagOptions(sub_tol_schedule = :eta2) isa AugLagOptions
    @test AugLagOptions(sub_tol_schedule = :fixed) isa AugLagOptions

    # end-to-end with the Ipopt sub-solver and the default (:eta) schedule:
    # the sub-solver options get the adaptive tolerance each outer iteration
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> [x[1] - 1.5, x[2] - 2.5], [0.0, 0.0])
    r = optimize(
        m,
        AugLagAlg(NonconvexIpopt.IpoptAlg()),
        [0.0, 0.0];
        options = AugLagOptions(
            max_iter = 50,
            sub_options = NonconvexIpopt.IpoptOptions(max_iter = 200),
        ),
    )
    g = m.ineq_constraints.fs[1](r.minimizer)
    @test maximum(max.(g, 0.0)) < 1e-4
    @test r.minimum < 1.0 + 1e-4
end
