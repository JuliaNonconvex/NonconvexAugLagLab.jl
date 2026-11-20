@testitem "plain ALM with exact Hessians on a JuMP model" begin
    using Test, NonconvexCore, NonconvexAugLagLab
    using JuMP: JuMP
    using NonconvexIpopt: NonconvexIpopt
    # Requires the lagrangian_hessian hook (JuliaNonconvex/NonconvexIpopt.jl#40).
    # On older NonconvexIpopt the hook argument is dropped silently and the
    # solve would run the corrupted nested-AD path; skip instead.
    :lagrangian_hessian in fieldnames(NonconvexIpopt.IpoptWorkspace) || return nothing

    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:2])
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 2)^2 + 0.0 * sin(x[1]))
    JuMP.@constraint(jm, x[1] + x[2] <= 5.0)  # inactive at the optimum
    JuMP.@constraint(jm, x[1] >= -1.0)
    dm = DictModel(jm)
    x0 = NonconvexCore.getinit(dm)
    r = optimize(
        dm,
        AugLagAlg(NonconvexIpopt.IpoptAlg()),
        x0;
        options=AugLagOptions(;
            max_iter=50,
            sub_options=NonconvexIpopt.IpoptOptions(
                max_iter=200, hessian_approximation="exact"
            ),
        ),
    )
    # the exact-Hessian subproblem solves run through the composed evaluator
    # Hessian (nested AD through the JuMP functions is rejected upstream)
    @test abs(dm.objective(r.minimizer)) < 1e-6
    g = dm.ineq_constraints.fs[1](r.minimizer)
    @test maximum(max.(g, 0.0)) < 1e-6
end

@testitem "the composed plain AL Hessian matches finite differences" begin
    using Test, NonconvexCore, NonconvexAugLagLab, LinearAlgebra, Random
    using JuMP: JuMP
    const NAL = NonconvexAugLagLab

    Random.seed!(0)
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:2])
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 2)^2 + x[1] * x[2] + 0.0 * sin(x[1]))
    JuMP.@constraint(jm, x[1] + x[2] <= 1.0)      # violated at the test point
    JuMP.@constraint(jm, x[1] - x[2] == 0.5)
    dm = DictModel(jm)
    vm, xv, _ = NonconvexCore.tovecmodel(dm, NonconvexCore.getinit(dm))
    je = NAL._jump_evaluator(vm.objective)
    @test je !== nothing
    eq_fs = vm.eq_constraints.fs
    ineq_fs = vm.ineq_constraints.fs
    @test all(NAL._is_jump_block(c, je) for c in eq_fs)
    @test all(NAL._is_jump_block(c, je) for c in ineq_fs)

    x = [3.0, 3.0]  # g = 5.0 > 0: the inequality is active, exercising J'J
    scales = ones(length(eq_fs) + length(ineq_fs))
    y = [randn(length(NAL._vecval(c(x)))) for c in eq_fs]
    λ = [rand(length(NAL._vecval(c(x)))) for c in ineq_fs]
    w_eq = ones(length(eq_fs))
    w_ineq = ones(length(ineq_fs))
    ρ = 3.7
    fscale = 1.0
    obj = vm.objective

    function al_obj(z)
        v = obj(z) / fscale
        for (k, c) in enumerate(eq_fs)
            cv = NAL._vecval(c(z))
            v += dot(y[k], cv) + (ρ / 2) * sum(abs2, cv)
        end
        for (k, c) in enumerate(ineq_fs)
            gv = max.(NAL._vecval(c(z)), 0.0)
            v += dot(λ[k], gv) + (ρ / 2) * sum(abs2, gv)
        end
        return v
    end
    H = NAL._jump_al_hessian(
        je, x, eq_fs, ineq_fs, scales, y, λ, w_eq, w_ineq, Ref(ρ), fscale
    )
    n = length(x)
    h = 1e-5
    Hfd = zeros(n, n)
    for i in 1:n, j in 1:n
        ei = zeros(n)
        ei[i] = 1.0
        ej = zeros(n)
        ej[j] = 1.0
        Hfd[i, j] =
            (
                al_obj(x + h * ei + h * ej) - al_obj(x + h * ei - h * ej) -
                al_obj(x - h * ei + h * ej) + al_obj(x - h * ei - h * ej)
            ) / (4h^2)
    end
    @test maximum(abs.(H - Hfd)) / maximum(abs.(Hfd)) < 1e-5
end
