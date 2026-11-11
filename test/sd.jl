using NonconvexAugLagLab, NonconvexIpopt, NonconvexSemidefinite, Test, LinearAlgebra

@testitem "AugLagAlg: semidefinite constraint (projected ALM)" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, Test, LinearAlgebra
    # min x² s.t. x ≥ 2 via a 1x1 PSD constraint x - 2 ≥ 0 → x = 2.
    m = Model(x -> x[1]^2)
    addvar!(m, [-5.0], [5.0])
    add_sd_constraint!(m, x -> [x[1] - 2.0;;])
    r = optimize(m, AugLagAlg(), [0.0])
    @test abs(r.minimizer[1] - 2.0) < 1e-4
    @test abs(r.minimum - 4.0) < 1e-4
end

@testitem "AugLagAlg: sd kept hard requires the sub-solver to support it" begin
    using NonconvexAugLagLab, NonconvexIpopt, NonconvexSemidefinite, Test, LinearAlgebra
    m = Model(x -> x[1]^2)
    addvar!(m, [-5.0], [5.0])
    add_sd_constraint!(m, x -> [x[1] - 2.0;;])
    options = AugLagOptions(
        relaxed=RelaxConstraintSelection(sd=false), sub_options=IpoptOptions(max_iter=10)
    )
    # Ipopt does not support sd constraints; the strict capability check errors.
    @test_throws ArgumentError optimize(m, AugLagAlg(IpoptAlg()), [0.0]; options=options)
end

@testitem "AugLagAlg: chordal-decomposed sd constraint (block evaluation)" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, Test, LinearAlgebra, SparseArrays

    # min (x₁ - 1)² + (x₂ - 1)² s.t. tridiag(x₁, x₂) ⪰ 0, chordally decomposed.
    P(x) = [x[1] 0.25 0.0 0.0; 0.25 x[2] 0.25 0.0; 0.0 0.25 x[1] 0.25; 0.0 0.0 0.25 x[2]]
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [0.0, 0.0], [5.0, 5.0])
    add_sd_constraint!(m, P)
    vm = NonconvexCore.tovecmodel(m)[1]
    dm = chordal_decompose(vm, sparse(P(vm.init) .!= 0))
    @test length(dm.sd_constraints.fs) == 1
    @test dm.sd_constraints.fs[1] isa NonconvexSemidefinite.BlockSDConstraint

    r = NonconvexCore.optimize(
        dm, AugLagAlg(), [2.0, 2.0]; options=AugLagOptions(ρ_init=100.0)
    )
    @test abs(r.minimizer[1] - 1.0) < 5e-2
    @test abs(r.minimizer[2] - 1.0) < 5e-2
    @test minimum(eigvals(Symmetric(P(r.minimizer)))) >= -1e-5
end
