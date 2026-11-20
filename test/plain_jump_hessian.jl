@testitem "plain ALM with exact Hessians on a JuMP model" begin
    using Test, NonconvexCore, NonconvexAugLagLab
    using JuMP: JuMP
    using NonconvexIpopt: NonconvexIpopt

    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:2])
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 2)^2)
    JuMP.@constraint(jm, x[1] + x[2] <= 5.0)  # inactive at the optimum
    JuMP.@constraint(jm, x[1] >= -1.0)
    dm = DictModel(jm)
    x0 = NonconvexCore.getinit(dm)
    r = optimize(
        dm,
        AugLagAlg(NonconvexIpopt.IpoptAlg()),
        x0;
        options = AugLagOptions(;
            max_iter = 50,
            sub_options = NonconvexIpopt.IpoptOptions(
                max_iter = 200, hessian_approximation = "exact"
            ),
        ),
    )
    # the exact-Hessian subproblem solves run through the composed evaluator
    # Hessian (nested AD through the JuMP functions is rejected upstream)
    @test abs(dm.objective(r.minimizer)) < 1e-6
    g = dm.ineq_constraints.fs[1](r.minimizer)
    @test maximum(max.(g, 0.0)) < 1e-6
end
