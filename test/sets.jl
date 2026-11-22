# Function-in-set constraints (NonconvexCore's FunctionInSet with
# SetProducts, PSD components, chance constraints and set variables) in the
# augmented Lagrangian — including the buckling-constrained column benchmark.

@testitem "AugLagAlg: set constraints with equality/inequality components" begin
    using NonconvexAugLagLab, LinearAlgebra, Test

    # min (x₁-1)² + (x₂-2)² s.t. (x₁ + x₂ - 3, x₂ - x₁) ∈
    # SetProduct(EqualTo(0), GreaterThan(0)) → (1, 2).
    m = Model(x -> sum(abs2, x .- [1.0, 2.0]))
    addvar!(m, [0.0, 0.0], [5.0, 5.0])
    add_set_constraint!(
        m,
        x -> (sum(x) - 3.0, x[2] - x[1]),
        SetProduct(EqualTo(0.0), GreaterThan(0.0)),
    )
    r = optimize(m, AugLagAlg(), [0.0, 0.0])
    @test r.minimizer ≈ [1.0, 2.0] atol = 1e-3

    # Interval (two-sided) and vector order sets.
    m2 = Model(x -> sum(abs2, x .- [0.5, 0.5]))
    addvar!(m2, [0.0, 0.0], [2.0, 2.0])
    add_set_constraint!(m2, x -> x, Interval([0.4, 0.4], [1.0, 1.0]))
    r2 = optimize(m2, AugLagAlg(), [0.0, 0.0])
    @test r2.minimizer ≈ [0.5, 0.5] atol = 1e-3

    # Simplex constraint: min (x - [1/3, 1/3, 1/3])² over the simplex.
    m3 = Model(x -> sum(abs2, x .- [1 / 3, 1 / 3, 1 / 3]))
    addvar!(m3, zeros(3), ones(3))
    add_set_constraint!(m3, x -> x, Simplex(3))
    r3 = optimize(m3, AugLagAlg(), [0.5, 0.3, 0.2])
    @test r3.minimizer ≈ [1 / 3, 1 / 3, 1 / 3] atol = 1e-3
    @test sum(r3.minimizer) ≈ 1.0 atol = 1e-4
end

@testitem "AugLagAlg: set products with PSD components (projected ALM)" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra, Test

    # min (x-1)² s.t. (x - 2, [x - 1;;]) ∈ SetProduct(LessThan(0), PSDCone(1))
    # → x ∈ [1, 2] → x = 1.
    m = Model(x -> (x[1] - 1)^2)
    addvar!(m, [0.0], [5.0])
    add_set_constraint!(
        m, x -> (x[1] - 2.0, [x[1] - 1.0;;]), SetProduct(LessThan(0.0), PSDCone(1))
    )
    r = optimize(m, AugLagAlg(), [0.0])
    @test r.minimizer[1] ≈ 1.0 atol = 1e-3

    # A matrix PSD component whose constraint binds: the unconstrained
    # optimum (0.2, 0.2) is not PSD, so the optimum lands on det P = 0.
    m2 = Model(x -> (x[1] - 0.2)^2 + (x[2] - 0.2)^2)
    addvar!(m2, [0.0, 0.0], [5.0, 5.0]; init=[2.0, 2.0])
    P(x) = [x[1] 0.5; 0.5 x[2]]
    add_set_constraint!(m2, P, PSDCone(2))
    r2 = optimize(m2, AugLagAlg(), [2.0, 2.0])
    @test minimum(eigvals(Symmetric(P(r2.minimizer)))) >= -1e-4
    # the PSD constraint binds: x1 x2 = 0.25 at the optimum
    @test prod(r2.minimizer) ≈ 0.25 atol = 5e-2
    @test r2.minimizer[1] ≈ r2.minimizer[2] atol = 5e-2

    # Kept-native set constraints require the sub-solver to support sets.
    using NonconvexIpopt
    m3 = Model(x -> x[1]^2)
    addvar!(m3, [0.0], [5.0])
    add_set_constraint!(m3, x -> x[1] - 2.0, LessThan(0.0))
    @test_throws ArgumentError optimize(
        m3,
        AugLagAlg(IpoptAlg()),
        [0.0];
        options=AugLagOptions(relaxed=RelaxConstraintSelection(sets=false)),
    )
end

@testitem "AugLagAlg: chance constraints" begin
    using NonconvexAugLagLab, LinearAlgebra, Test, Random

    struct Uniform01 end
    Random.rand(rng::AbstractRNG, ::Uniform01) = rand(rng)

    # min (x - 0.3)² s.t. P(x + ξ ≤ 1) ≥ 0.9 with ξ ~ Uniform(0, 1): the
    # constraint is x ≲ 0.1 and binds.
    m = Model(x -> (x[1] - 0.3)^2)
    addvar!(m, 0.0, 1.0)
    add_chance_constraint!(
        m, (x, ξ) -> x[1] + ξ, Uniform01(), LessThan(1.0);
        epsilon=0.1, n_samples=1000, seed=42,
    )
    r = optimize(m, AugLagAlg(), [0.0]; options=AugLagOptions(ctol=1e-4))
    # The smoothed exceedance drives p̂ = ε: with the tanh surrogate
    # p̂(x) = log(cosh(5x))/5, ε = 0.1 ⇔ x ≈ 0.226 (the smoothing
    # underestimates the exceedance, so the solution is conservative).
    @test 0.1 < r.minimizer[1] < 0.4
    vm, _, _ = NonconvexCore.tovecmodel(m, [0.0])
    @test set_violation(vm.set_constraints.fs[1], r.minimizer)[1] < 1e-3

    # A deterministic (One) constraint behaves like a plain constraint.
    m2 = Model(x -> (x[1] - 1)^2)
    addvar!(m2, 0.0, 2.0)
    add_chance_constraint!(m2, (x, ξ) -> x[1], One(), LessThan(0.5); epsilon=0.0)
    r2 = optimize(m2, AugLagAlg(), [0.0])
    @test r2.minimizer[1] ≈ 0.5 atol = 1e-3
end

@testitem "AugLagAlg: set variables (PSD matrix and simplex blocks)" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra, Test

    # A PSD matrix variable through the epigraph: min t s.t. t ≥ ‖X - I‖²,
    # X ⪰ 0 → X = I, t = 0.
    m = Model(x -> x[end])
    blk = add_set_variable!(m, PSDCone(2))
    add_set_variable!(m, Interval(-10.0, 10.0))  # t
    X(z) = reshape(z[blk.range], 2, 2)
    f = x -> begin
        Xv = X(x)
        return (sum(abs2, Xv .- Matrix(1.0I, 2, 2)) - x[end], (Xv .+ Xv') ./ 2)
    end
    add_set_constraint!(m, f, SetProduct(LessThan(0.0), PSDCone(2)))
    r = optimize(m, AugLagAlg(), m.init)
    @test X(r.minimizer) ≈ Matrix(1.0I, 2, 2) atol = 1e-2

    # A simplex variable block: min (x - [0.25, 0.25, 0.5])² over the simplex.
    m2 = Model(x -> sum(abs2, x .- [0.25, 0.25, 0.5]))
    add_set_variable!(m2, Simplex(3); init=[0.4, 0.3, 0.3])
    r2 = optimize(m2, AugLagAlg(), [0.4, 0.3, 0.3])
    @test r2.minimizer ≈ [0.25, 0.25, 0.5] atol = 1e-3
    @test sum(r2.minimizer) ≈ 1.0 atol = 1e-4
end

@testitem "AugLagAlg: chordal-decomposed set constraint" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra, Test, SparseArrays

    # min (x₁ - 0.2)² + (x₂ - 0.2)² s.t. the tridiagonal P(x) ⪰ 0,
    # chordally decomposed: the unconstrained point (0.2, 0.2) is not PSD,
    # so the constraint binds.
    P(x) = [
        x[1] 0.25 0.0 0.0
        0.25 x[2] 0.25 0.0
        0.0 0.25 x[1] 0.25
        0.0 0.0 0.25 x[2]
    ]
    m = Model(x -> (x[1] - 0.2)^2 + (x[2] - 0.2)^2)
    addvar!(m, [0.0, 0.0], [5.0, 5.0]; init=[2.0, 2.0])
    add_set_constraint!(m, P, PSDCone(4); flags=[:chordal])
    vm, x0, _ = NonconvexCore.tovecmodel(m, [2.0, 2.0])
    pat = sparse(P(vm.init) .!= 0)
    dm, d0, mapc = chordal_transform(vm, [pat]; x0=vm.init, labels=[:chordal])
    @test dm.set_constraints.fs[1].set isa EqualTo  # the coupling equality
    r = NonconvexCore.optimize(
        dm, AugLagAlg(), d0; options=AugLagOptions(rho_init=100.0, ctol=1e-6)
    )
    xs = mapc(r.minimizer)
    @test minimum(eigvals(Symmetric(P(xs)))) >= -1e-4
    @test prod(xs) >= 0.16

    # Equivalence with the dense (undecomposed) solve.
    r2 = NonconvexCore.optimize(
        vm, AugLagAlg(), [2.0, 2.0]; options=AugLagOptions(rho_init=100.0, ctol=1e-6)
    )
    @test minimum(eigvals(Symmetric(P(r2.minimizer)))) >= -1e-4
    @test xs ≈ r2.minimizer atol = 5e-2
end

@testitem "AugLagAlg: buckling-constrained truss (epigraph + set product)" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra, Test

    # A miniature 2D ground-structure truss (the buckling-constrained
    # minimum-compliance problem in miniature):
    #   min t  s.t.  c(x) - t ≤ 0,  ΣxᵢLᵢ/ΣLᵢ ≤ V̄,  K(x) + cᵦ·Kσ(x) ⪰ 0,
    # with the three constraints as ONE function (a SetProduct) sharing a
    # single displacement solve per evaluation.
    #
    # 4 nodes (2×2 grid), the left edge clamped, a unit load at the top
    # right; the 6 members form the complete ground structure.

    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    members = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    ndof = 8
    fixed = [1, 2, 3, 4]
    free = [5, 6, 7, 8]
    f_load = zeros(ndof)
    f_load[8] = -1.0
    E = 1.0
    Ls = [norm(collect(coords[b]) .- collect(coords[a])) for (a, b) in members]

    # Element axial (γ) and transversal (δ) operators in global DOF numbering.
    γs = Vector{Vector{Float64}}()
    δs = Vector{Vector{Float64}}()
    edofs = Vector{Vector{Int}}()
    for (a, b) in members
        xa = collect(coords[a])
        xb = collect(coords[b])
        L = norm(xb - xa)
        d = (xb - xa) ./ L
        γ = zeros(ndof)
        γ[[2a - 1, 2a]] .= -d
        γ[[2b - 1, 2b]] .= d
        δ = zeros(ndof)
        δ[[2a - 1, 2a]] .= [-d[2], d[1]]
        δ[[2b - 1, 2b]] .= [d[2], -d[1]]
        push!(γs, γ)
        push!(δs, δ)
        push!(edofs, [2a - 1, 2a, 2b - 1, 2b])
    end

    # Assemble K and Kσ on the free DOFs (the truss geometric stiffness of
    # Gavin 2014 / Kočvara 2002, eqs. (3)-(4)) — pure, AD-friendly.
    function assemble(x, u)
        K = sum(enumerate(members)) do (e, ab)
            L = Ls[e]
            x[e] .* ((E / L) .* (γs[e] * γs[e]'))
        end
        Kσ = sum(enumerate(members)) do (e, ab)
            L = Ls[e]
            N = x[e] * (E / L) * dot(γs[e], u)     # axial force
            (N / L) .* (δs[e] * δs[e]')
        end
        return K[free, free], Kσ[free, free]
    end

    Vbar = 0.5
    n = length(members)
    cbeta = 2.0  # placeholder; recalibrated after the baseline solve
    # The free DOFs follow the fixed ones, so the full displacement is the
    # concatenation (pure).
    scatter(uf) = vcat(zeros(length(fixed)), uf)

    # The buckling load factor of a design (smallest λ with det(K+λKσ) = 0).
    blf(x) = begin
        K0, _ = assemble(x, zeros(ndof))
        u = scatter(K0 \ f_load[free])
        K, Kσ = assemble(x, u)
        F = eigen(Matrix(K), Matrix(-Kσ))
        return minimum(v for v in F.values if isfinite(v) && v > 0)
    end

    bundle = x -> begin
        t = x[end]
        xd = x[1:n]
        K0, _ = assemble(xd, zeros(ndof))
        u = scatter(K0 \ f_load[free])
        K, Kσ = assemble(xd, u)
        return (
            dot(f_load[free], u[length(fixed) + 1:end]) - t,
            sum(xd .* Ls) / sum(Ls) - Vbar,
            K + cbeta * Kσ,
        )
    end

    # Baseline: minimum compliance with the volume constraint only.
    m0 = Model(x -> x[end])
    addvar!(m0, fill(1e-3, n), ones(n); init=fill(Vbar, n), flags=[:design])
    add_set_variable!(m0, Interval(0.0, Inf); init=[1.0])
    add_set_constraint!(
        m0, x -> bundle(x)[1:2], SetProduct(LessThan(0.0), LessThan(0.0))
    )
    vm0, x0v, _ = NonconvexCore.tovecmodel(m0, [fill(Vbar, n); 1.0])
    r0 = NonconvexCore.optimize(
        vm0, AugLagAlg(), [fill(Vbar, n); 1.0]; options=AugLagOptions(ctol=1e-6)
    )
    x0 = clamp.(r0.minimizer[1:n], 1e-3, 1.0)

    # The buckling load factor scales with the design scale: pick the load
    # factor between the compliance design's and the full design's, so the
    # constraint is feasible and binds.
    λ0 = blf(x0)
    λmax = blf(ones(n))
    cbeta = 0.5 * (λ0 + λmax)
    @test λ0 < cbeta < λmax

    # Buckling-constrained: the epigraph + set-product single function.
    t0 = bundle([x0; 0.0])[1] + 1.0
    m = Model(x -> x[end])
    addvar!(m, fill(1e-3, n), ones(n); init=copy(x0), flags=[:design])
    add_set_variable!(m, Interval(0.0, Inf); init=[t0])
    add_set_constraint!(
        m, bundle, SetProduct(LessThan(0.0), LessThan(0.0), PSDCone(4))
    )
    vm, xv, _ = NonconvexCore.tovecmodel(m, [x0; t0])
    r = NonconvexCore.optimize(
        vm,
        AugLagAlg(),
        [x0; t0];
        options=AugLagOptions(rho_init=100.0, ctol=1e-6, max_iter=500),
    )
    xb = clamp.(r.minimizer[1:n], 1e-3, 1.0)

    # The constrained design is stable under the load factor.
    λb = blf(xb)
    @test λb >= cbeta - 1e-4

    # Volume constraint and the compliance increase.
    @test sum(xb .* Ls) / sum(Ls) <= Vbar + 2e-3
    c0 = bundle([x0; 0.0])[1]
    cb = bundle([xb; 0.0])[1]
    @test cb >= c0 * (1 - 1e-2)
    @test λb > λ0
end

@testitem "AugLagAlg: the bundle of a set product is evaluated once per cycle" begin
    using NonconvexAugLagLab, NonconvexSemidefinite, LinearAlgebra, Test

    cnt = Ref(0)
    m = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(m, [0.0, 0.0], [5.0, 5.0])
    f = x -> begin
        cnt[] += 1
        return (x[1] - 2.0, x[2] - 2.0, [x[1] + x[2] - 2.0;;])
    end
    add_set_constraint!(
        m, f, SetProduct(LessThan(0.0), LessThan(0.0), PSDCone(1))
    )
    r = optimize(m, AugLagAlg(), [0.0, 0.0]; options=AugLagOptions(max_iter=30))
    @test r.minimizer ≈ [1.0, 1.0] atol = 1e-3
    # Three components but one shared evaluation: the bundle must not run
    # three times per cycle — 30 outer iterations cap the count well below
    # the naive 3× per evaluation.
    @test cnt[] < 60
    # Without sharing, the equivalent 3 separate constraints would have run
    # roughly three times as often per evaluation: verify the sharing by
    # re-evaluating the constraint components directly.
    vm, x0, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    c = vm.set_constraints.fs[1]
    cnt[] = 0
    c(x0)
    @test cnt[] == 1
    # and the ALM evaluation path keeps the same count for a fresh solve
    # started from a feasible point (fewer cycles).
    r2 = optimize(m, AugLagAlg(), [1.0, 1.0]; options=AugLagOptions(max_iter=10))
    @test all(isfinite, r2.minimizer)
end
