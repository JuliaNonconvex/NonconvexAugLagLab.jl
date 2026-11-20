@testitem "built-in QN honors sub_options.tol" begin
    using Test, NonconvexCore, NonconvexAugLagLab

    # Regression: sub_options.tol used to be silently dropped for the
    # built-in quasi-Newton sub-solver (only max_iter/globalization/etc.
    # were read), so a user-provided inner tolerance had no effect and the
    # schedule always applied. The override must now take precedence and
    # the solve must converge for loose and tight tolerances alike.
    m = Model(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, x -> [x[1] - 1.5, x[2] - 2.5], [0.0, 0.0])
    for tol in (1e-2, 1e-6, 1e-9)
        r = optimize(
            m,
            AugLagAlg(),
            [0.0, 0.0];
            options = AugLagOptions(;
                max_iter = 100,
                sub_tol_schedule = :fixed,
                sub_options = (; max_iter = 200, tol = tol, globalization = :trustregion),
            ),
        )
        @test r.minimum < 1.0 + 1e-4
        g = m.ineq_constraints.fs[1](r.minimizer)
        @test maximum(max.(g, 0.0)) < 1e-4
    end
end
