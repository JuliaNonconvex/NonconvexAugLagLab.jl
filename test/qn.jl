using NonconvexAugLagLab, Test, LinearAlgebra, SparseArrays

@testitem "QuasiNewtonAlg: L-BFGS on the Rosenbrock function" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(), [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    @test abs(r.minimum) < 1e-6
end

@testitem "QuasiNewtonAlg: L-BFGS with a diagonal preconditioner" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    alg = QuasiNewtonAlg(LBFGSApprox(10; H0=DiagonalPreconditioner([1.0, 100.0])))
    r = optimize(m, alg, [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
end

@testitem "QuasiNewtonAlg: dense BFGS inverse (default) and Hessian form" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    for form in (:inverse, :hessian)
        m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        r = optimize(m, QuasiNewtonAlg(DenseBFGSApprox(form=form)), [-1.2, 1.0])
        @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
    end
end

@testitem "QuasiNewtonAlg: structured sparse BFGS" begin
    using NonconvexAugLagLab, Test, LinearAlgebra, SparseArrays
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    # diagonal sparsity pattern: only the diagonal of the Hessian model is kept
    B = spdiagm(ones(2))
    r = optimize(m, QuasiNewtonAlg(StructuredBFGSApprox(B)), [-1.2, 1.0]; options=nothing)
    @test norm(r.minimizer - [1.0, 1.0]) < 5e-2
end

@testitem "QuasiNewtonAlg: Newton trust region" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    m = Model(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(NewtonTRApprox(); linesearch=:trustregion), [-1.2, 1.0])
    @test norm(r.minimizer - [1.0, 1.0]) < 1e-3
end

@testitem "QuasiNewtonAlg: user-defined Hessian approximation" begin
    using NonconvexAugLagLab, Test, LinearAlgebra

    # A custom Hessian approximation through the extension API: a fixed diagonal
    # scaling that accumulates nothing.
    struct FixedDiagonalApprox <: AbstractHessianApprox
        d::Vector{Float64}
    end
    function NonconvexAugLagLab.init!(H::FixedDiagonalApprox, n, x0)
        return H
    end
    function NonconvexAugLagLab.apply!(Hv, H::FixedDiagonalApprox, v)
        Hv .= H.d .* v
        return Hv
    end
    function NonconvexAugLagLab.update!(H::FixedDiagonalApprox, s, y)
        return H
    end
    function NonconvexAugLagLab.solve!(d, H::FixedDiagonalApprox, g)
        d .= g ./ H.d
        return d
    end

    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = optimize(m, QuasiNewtonAlg(FixedDiagonalApprox([1.0, 1.0])), [3.0, 4.0])
    @test norm(r.minimizer) < 1e-3
end

@testitem "QuasiNewtonAlg: L-BFGS apply! is the direct Hessian matvec" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # Fixed SPD quadratic Hessian. With A-conjugate secant pairs (y = A s,
    # s₁ᵀAs₂ = 0) BFGS reproduces A exactly after the two updates, so the
    # L-BFGS compact direct form, the dense :hessian matvec and A * v must
    # all agree.
    A = [2.0 0.5; 0.5 1.0]
    s1 = [1.0, 0.0]
    s2 = [1.0, -4.0]   # s2ᵀ * A * s1 = 0
    @test dot(s2, A * s1) == 0.0
    pairs = [(s1, A * s1), (s2, A * s2)]
    dense = DenseBFGSApprox(form=:hessian)
    NonconvexAugLagLab.init!(dense, 2, nothing)
    lb = LBFGSApprox(5)
    NonconvexAugLagLab.init!(lb, 2, nothing)
    v = [0.7, -1.1]
    Bv_dense = similar(v)
    Bv_lb = similar(v)
    for (s, y) in pairs
        NonconvexAugLagLab.update!(dense, s, y)
        NonconvexAugLagLab.update!(lb, s, y)
        NonconvexAugLagLab.apply!(Bv_dense, dense, v)
        NonconvexAugLagLab.apply!(Bv_lb, lb, v)
        @test Bv_lb ≈ Bv_dense   # matches the dense BFGS :hessian matvec
    end
    @test Bv_lb ≈ A * v          # and the true Hessian product
    # with the identity H0 the direct product differs from the inverse solve
    d = similar(v)
    NonconvexAugLagLab.solve!(d, lb, v)
    @test !(d ≈ Bv_lb)
    @test d ≈ A \ v
    # DenseBFGS form=:inverse stores B⁻¹: apply! must solve with it, so the
    # direct matvec still equals A v
    dinv = DenseBFGSApprox(form=:inverse)
    NonconvexAugLagLab.init!(dinv, 2, nothing)
    for (s, y) in pairs
        NonconvexAugLagLab.update!(dinv, s, y)
    end
    Binv_v = similar(v)
    NonconvexAugLagLab.apply!(Binv_v, dinv, v)
    @test Binv_v ≈ A * v
end

@testitem "QuasiNewtonAlg: TRON projected trust region" begin
    using NonconvexAugLagLab, Test, LinearAlgebra
    # Box-constrained quadratic with the optimum on a bound.
    m = Model(x -> (x[1] - 1)^2 + 2 * (x[2] - 2)^2)
    addvar!(m, [0.0, 0.0], [1.0, Inf])
    r = optimize(m, QuasiNewtonAlg(; linesearch=:tron, max_iter=500), [0.1, 0.1])
    @test norm(r.minimizer - [1.0, 2.0]) < 1e-4
    # Nonconvex objective with a saddle at the origin.
    m2 = Model(x -> x[1]^4 - 2 * x[1]^2)
    addvar!(m2, [-2.0], [2.0])
    r2 = optimize(m2, QuasiNewtonAlg(; linesearch=:tron, max_iter=500), [0.1])
    @test abs(abs(r2.minimizer[1]) - 1.0) < 1e-3
    @test abs(r2.minimum + 1.0) < 1e-4
    # The AL wrapper accepts :tron through sub_options.
    m3 = Model(x -> sum(x .^ 2))
    addvar!(m3, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m3, x -> 2 - x[1] - x[2])
    r3 = optimize(
        m3,
        AugLagAlg(),
        [1.5, 1.5];
        options=AugLagOptions(sub_options=(; max_iter=200, tol=1e-6, linesearch=:tron)),
    )
    @test norm(r3.minimizer - [1.0, 1.0]) < 1e-3
end
