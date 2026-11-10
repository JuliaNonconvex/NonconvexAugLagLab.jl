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
