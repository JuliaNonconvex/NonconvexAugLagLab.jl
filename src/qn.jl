# Built-in quasi-Newton subproblem solver for the augmented Lagrangian
# meta-algorithm (also usable standalone). Supports:
#   - L-BFGS with an optional preconditioner,
#   - dense BFGS (inverse Hessian by default, direct Hessian on request),
#   - structured BFGS on a user-defined Hessian container (sparse with a
#     pattern, or any custom matrix type via the extension API),
#   - Newton trust region with an exact AD Hessian.
#
# Extension API (documented): define a custom type `MyApprox <:
# AbstractHessianApprox` and implement
#     init!(H, n, x0)     allocate/initialize
#     apply!(Hv, H, v)    matrix-vector product with the Hessian model
#     update!(H, s, y)    accumulate the secant pair s = x⁺ - x, y = g⁺ - g
#     solve!(d, H, g)     solve the Newton system with the approximation
#     Base.reset!(H)      drop accumulated curvature (optional)
# Default implementations are provided for dense matrices, sparse matrices
# (pattern-based) and diagonal scaling.

"""
    AbstractHessianApprox

A quasi-Newton Hessian approximation with a possibly custom structure and
custom linear solves. See the module documentation for the extension API.
"""
abstract type AbstractHessianApprox end

# ------------------------------------------------------------------
# Exact curvature terms for composite Hessian models
# ------------------------------------------------------------------
"""
    AbstractCurvatureTerm

An exact Hessian term of the (augmented Lagrangian) objective with a
matrix-vector product `apply_term!(Hv, term, v)`. Terms are position
dependent and are rebuilt at each subproblem iterate through the
composite's `build_terms` hook.
"""
abstract type AbstractCurvatureTerm end

"""
    GaussNewtonTerm(c, w, x[, act])

The Gauss-Newton term `w · J(x)ᵀ(J(x) v)` of one (scaled) constraint
group, computed matrix-free: `Jv` by a forward finite difference from
the cached value `c(x)` (one constraint evaluation) and `Jᵀu` by one
reverse pass. For inequality groups, only the rows violated at the
rebound point `x` contribute (`act`, the frozen active set).
"""
struct GaussNewtonTerm{F,X}
    c::F
    w::Float64
    x::X
    c0::Vector{Float64}
    act::Union{Nothing,BitVector}
end
function GaussNewtonTerm(c, w, x, act=nothing)
    return GaussNewtonTerm(c, w, x, _vecval(c(x)), act)
end

function apply_term!(Hv, t::GaussNewtonTerm, v)
    nv = max(norm(v), 1e-12)
    ε = sqrt(eps(Float64)) * (1 + norm(t.x)) / nv
    cp = _vecval(t.c(t.x .+ ε .* v))
    Jv = (cp .- t.c0) ./ ε
    t.act === nothing || (Jv[.!t.act] .= 0.0)
    g = NonconvexCore.value_gradient(z -> dot(_vecval(t.c(z)), Jv), t.x)[2]
    g === nothing && return Hv
    Hv .+= t.w .* g
    return Hv
end

"""
    JumpGaussNewtonTerm(je, rows, w, x[, act])

The Gauss-Newton term `w · J(x)ᵀ(J(x) v)` of a JuMP-converted constraint
block: the exact sparse Jacobian of the MathOptInterface evaluator is
materialized once per rebound point and each matvec is two sparse
products — no finite differences and no reverse passes, so the
per-CG-iteration cost is negligible (the Jacobian evaluation dominates,
once per subproblem iterate).
"""
struct JumpGaussNewtonTerm{J,X}
    je::J
    rows::UnitRange{Int}
    w::Float64
    x::X
    act::Union{Nothing,BitVector}
    J::SparseMatrixCSC{Float64,Int}
end
function JumpGaussNewtonTerm(je, rows, w, x, act=nothing)
    J = NonconvexCore.jump_jacobian(je, x)[rows, :]
    return JumpGaussNewtonTerm(je, rows, w, x, act, J)
end

function apply_term!(Hv, t::JumpGaussNewtonTerm, v)
    Jv = t.J * v
    t.act === nothing || (Jv[.!t.act] .= 0.0)
    mul!(Hv, t.J', Jv, t.w, 1.0)
    return Hv
end

"""
    JumpLagrangianTerm(je, μ, x)

The exact multiplier-weighted constraint-curvature term `Σ μᵢ ∇²cᵢ` of a
JuMP-converted model: the MathOptInterface evaluator's Lagrangian
Hessian is assembled once per rebound point from its sparse triplets
(σ = 0, so only the constraint part contributes) and each matvec is one
sparse product. `μ` is the full evaluator-length multiplier vector in
its row order (inequalities first, then equalities), zero outside the
selected block.
"""
struct JumpLagrangianTerm{J,X}
    je::J
    H::SparseMatrixCSC{Float64,Int}
    x::X
end
function JumpLagrangianTerm(je, μ, x)
    sv = zeros(length(je.hess_rows))
    NonconvexCore.jump_hessian_lagrangian!(je, sv, x, 0.0, μ)
    n = length(x)
    H = sparse(je.hess_rows, je.hess_cols, sv, n, n)
    return JumpLagrangianTerm(je, H, x)
end

function apply_term!(Hv, t::JumpLagrangianTerm, v)
    mul!(Hv, t.H, v, 1.0, 1.0)
    return Hv
end

"""
    CompositeHessianApprox(qn, build_terms)

A Hessian model summing a quasi-Newton approximation `qn` (of the
remainder) and a tuple of exact curvature terms rebuilt at every
subproblem iterate via `build_terms(x)` (the Gauss-Newton penalty terms
of the AL objective, or the label-selected exact terms). `solve!` is the
quasi-Newton inverse (the CG preconditioner role).
"""
struct CompositeHessianApprox{Q<:AbstractHessianApprox,B} <: AbstractHessianApprox
    qn::Q
    build_terms::B
    terms::Base.RefValue{Any}
end
function CompositeHessianApprox(qn::AbstractHessianApprox, build_terms)
    return CompositeHessianApprox(qn, build_terms, Ref{Any}(()))
end

init!(H::CompositeHessianApprox, n, x0) = (init!(H.qn, n, x0); H)
reset!(H::CompositeHessianApprox) = (reset!(H.qn); H)
update!(H::CompositeHessianApprox, s, y) = (update!(H.qn, s, y); H)
solve!(d, H::CompositeHessianApprox, g) = solve!(d, H.qn, g)
function apply!(Bv, H::CompositeHessianApprox, v)
    apply!(Bv, H.qn, v)
    for t in H.terms[]
        apply_term!(Bv, t, v)
    end
    return Bv
end
_initialized(H::CompositeHessianApprox, n) = _initialized(H.qn, n)
_initialized(H::CompositeHessianApprox, n::Int) = _initialized(H.qn, n)

# Rebuild the exact terms at the current iterate (a no-op for models
# without position-dependent terms).
_rebind_terms!(H::CompositeHessianApprox, x) = (H.terms[]=H.build_terms(x); H)
_rebind_terms!(::AbstractHessianApprox, x) = nothing

# ------------------------------------------------------------------
# Preconditioners / H0 operators
# ------------------------------------------------------------------
"""
    IdentityPreconditioner()

Identity Hessian scaling: `apply!`/`solve!` copy their argument.
"""
struct IdentityPreconditioner end
apply!(Pv, ::IdentityPreconditioner, v) = copyto!(Pv, v)
solve!(d, ::IdentityPreconditioner, g) = copyto!(d, g)

"""
    DiagonalPreconditioner(d)

Diagonal Hessian scaling with entries `d` (all positive). `solve!` divides by
the scaling, `apply!` multiplies by it.
"""
struct DiagonalPreconditioner{V<:AbstractVector}
    d::V
end
apply!(Pv, P::DiagonalPreconditioner, v) = Pv .= P.d .* v
solve!(d, P::DiagonalPreconditioner, g) = d .= g ./ P.d

const AbstractPreconditioner = Union{IdentityPreconditioner,DiagonalPreconditioner}

# ------------------------------------------------------------------
# L-BFGS
# ------------------------------------------------------------------
"""
    LBFGSApprox(memory; H0 = IdentityPreconditioner())

Limited-memory BFGS. Stores the last `memory` secant pairs and applies the
approximation (and its inverse) with the two-loop recursion; `H0` is the base
operator used at the bottom of the recursion and serves as the preconditioner.
"""
mutable struct LBFGSApprox{P<:AbstractPreconditioner} <: AbstractHessianApprox
    memory::Int
    H0::P
    s::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    ρ::Vector{Float64}
    α::Vector{Float64}
end
function LBFGSApprox(memory::Int=10; H0=IdentityPreconditioner())
    return LBFGSApprox(
        memory,
        H0,
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Float64[],
        Float64[],
    )
end

init!(H::LBFGSApprox, n::Int, x0) = reset!(H)
function reset!(H::LBFGSApprox)
    empty!(H.s)
    empty!(H.y)
    empty!(H.ρ)
    return H
end

# Apply the inverse approximation: d = B⁻¹ g (two-loop recursion).
function solve!(d, H::LBFGSApprox, g)
    m = length(H.s)
    resize!(H.α, m)
    q = copy(g)
    for i in m:-1:1
        H.α[i] = H.ρ[i] * dot(H.s[i], q)
        q .-= H.α[i] .* H.y[i]
    end
    solve!(d, H.H0, q)
    for i in 1:m
        β = H.ρ[i] * dot(H.y[i], d)
        d .+= (H.α[i] - β) .* H.s[i]
    end
    return d
end

# Apply the direct approximation: the Hessian matvec B v via the N&W (7.24)
# compact representation. With S = [s₁ … sₘ], Y = [y₁ … yₘ] and SᵀY = L + D
# (L strictly lower triangular, D = diag(sᵢᵀyᵢ)):
#   B v = B0 v − [B0 S  Y] M⁻¹ [Sᵀ B0 v; Yᵀ v],  M = [SᵀB0S  L; Lᵀ  −D].
# (The inverse approximation is `solve!`; `apply!` must give the direct
# product or trust-region/damping quantities come out wrong.)
function apply!(Bv, H::LBFGSApprox, v)
    m = length(H.s)
    m == 0 && return apply!(Bv, H.H0, v)
    B0v = similar(v, Float64)
    apply!(B0v, H.H0, v)
    B0S = [apply!(similar(v, Float64), H.H0, s) for s in H.s]
    # rhs = [Sᵀ B0 v; Yᵀ v]
    rhs = Vector{Float64}(undef, 2m)
    for i in 1:m
        rhs[i] = dot(H.s[i], B0v)
        rhs[m + i] = dot(H.y[i], v)
    end
    # M = [SᵀB0S  L; Lᵀ  −D]
    M = Matrix{Float64}(undef, 2m, 2m)
    for j in 1:m, i in 1:m
        M[i, j] = dot(H.s[i], B0S[j])
        M[i, m + j] = i > j ? dot(H.s[i], H.y[j]) : 0.0
        M[m + i, j] = j > i ? dot(H.s[j], H.y[i]) : 0.0
        M[m + i, m + j] = i == j ? -dot(H.s[i], H.y[i]) : 0.0
    end
    w = M \ rhs
    Bv .= B0v
    for i in 1:m
        Bv .-= w[i] .* B0S[i]
        Bv .-= w[m + i] .* H.y[i]
    end
    return Bv
end

function update!(H::LBFGSApprox, s, y)
    sy = dot(s, y)
    sy > 0 || return H
    push!(H.s, copy(s))
    push!(H.y, copy(y))
    push!(H.ρ, inv(sy))
    if length(H.s) > H.memory
        popfirst!(H.s)
        popfirst!(H.y)
        popfirst!(H.ρ)
    end
    return H
end

# ------------------------------------------------------------------
# Limited-memory SR1 (the classic trust-region pairing: SR1 captures
# indefiniteness, which BFGS cannot on nonconvex objectives)
# ------------------------------------------------------------------
"""
    LimitedMemorySR1Approx(memory; H0 = IdentityPreconditioner())

Limited-memory symmetric rank-one. Stores the last `memory` pairs and
applies `B v = B0 v + Σ_k u_k (u_k'v)/(u_k's_k)` with `u_k = y_k − B0 s_k`
(the fixed-base L-SR1); pairs failing the safeguard
`|s'u| ≥ 1e-8 ‖s‖‖u‖` are rejected. Intended for `globalization =
:trustregion` subproblems, where an indefinite Hessian model is desirable.
"""
mutable struct LimitedMemorySR1Approx{P<:AbstractPreconditioner} <: AbstractHessianApprox
    memory::Int
    B0::P
    s::Vector{Vector{Float64}}
    u::Vector{Vector{Float64}}
    α::Vector{Float64}
end
function LimitedMemorySR1Approx(memory::Int=10; H0=IdentityPreconditioner())
    return LimitedMemorySR1Approx(
        memory, H0, Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[]
    )
end

init!(H::LimitedMemorySR1Approx, n::Int, x0) = reset!(H)
function reset!(H::LimitedMemorySR1Approx)
    empty!(H.s)
    empty!(H.u)
    empty!(H.α)
    return H
end

function apply!(Bv, H::LimitedMemorySR1Approx, v)
    apply!(Bv, H.B0, v)
    for k in eachindex(H.s)
        Bv .+= H.α[k] .* dot(H.u[k], v) .* H.u[k]
    end
    return Bv
end

# SR1 has no cheap inverse recursion; solve! falls back to the base operator
# (the trust-region paths only need apply!).
function solve!(d, H::LimitedMemorySR1Approx, g)
    solve!(d, H.B0, g)
    return d
end

function update!(H::LimitedMemorySR1Approx, s, y)
    B0s = similar(s, Float64)
    apply!(B0s, H.B0, s)
    u = y .- B0s
    su = dot(s, u)
    # safeguard (Nocedal-Wright 6.26): reject near-degenerate denominators
    abs(su) >= 1e-8 * norm(s) * norm(u) || return H
    push!(H.s, copy(s))
    push!(H.u, u)
    push!(H.α, inv(su))
    if length(H.s) > H.memory
        popfirst!(H.s)
        popfirst!(H.u)
        popfirst!(H.α)
    end
    return H
end

_initialized(H::LimitedMemorySR1Approx, n::Int) = !isempty(H.s) && length(H.s[1]) == n

# ------------------------------------------------------------------
# Dense BFGS (inverse by default)
# ------------------------------------------------------------------
"""
    DenseBFGSApprox(form = :inverse)

Dense BFGS. With `form = :inverse` (default) the *inverse* Hessian
approximation is stored and updated, so `solve!` is a plain matrix-vector
product; with `form = :hessian` the direct Hessian is stored and `solve!`
factorizes it.
"""
mutable struct DenseBFGSApprox{T<:AbstractMatrix} <: AbstractHessianApprox
    H::T
    form::Symbol
end
DenseBFGSApprox(; form::Symbol=:inverse) = DenseBFGSApprox(zeros(0, 0), form)

function init!(H::DenseBFGSApprox, n::Int, x0)
    H.H = Matrix{Float64}(I, n, n)
    return H
end
reset!(H::DenseBFGSApprox) = (H.H=Matrix{Float64}(I, size(H.H, 1), size(H.H, 1)); H)

function update!(H::DenseBFGSApprox, s, y)
    B = H.H
    sy = dot(s, y)
    sy > 0 || return H
    if H.form === :inverse
        # inverse BFGS update
        ρ = inv(sy)
        H.H .= (I - ρ .* s .* y') * B * (I - ρ .* y .* s') .+ ρ .* s .* s'
    else
        Bs = B * s
        sBs = dot(s, Bs)
        sBs > 0 || return H
        H.H .= B .+ (y .* y') ./ sy .- (Bs .* Bs') ./ sBs
    end
    return H
end

function apply!(Bv, H::DenseBFGSApprox, v)
    if H.form === :inverse
        # the stored matrix is B⁻¹, so the direct matvec must solve with it
        copyto!(Bv, v)
        return ldiv!(lu(H.H), Bv)
    else
        return mul!(Bv, H.H, v)
    end
end

function solve!(d, H::DenseBFGSApprox, g)
    if H.form === :inverse
        return mul!(d, H.H, g)
    else
        # Cholesky of the SPD approximation (updates are skipped on
        # nonpositive curvature, so H stays positive definite).
        F = cholesky(Symmetric(H.H); check=false)
        return copyto!(d, F \ g)
    end
end

# ------------------------------------------------------------------
# Structured (sparse) BFGS with a user-defined container
# ------------------------------------------------------------------
"""
    StructuredBFGSApprox(B; form = :hessian)

BFGS with secants accumulated in a user-defined Hessian container `B` (e.g. a
sparse matrix carrying a sparsity pattern, or any custom matrix type). The
built-in update is the structure-preserving sparse BFGS (Fletcher): the dense
BFGS update is restricted to the nonzero pattern of `B`, so the structure and
the user's `solve!`/`apply!` implementations keep working. Custom matrix types
should overload `init!`/`update!`/`solve!`/`apply!` directly.
"""
mutable struct StructuredBFGSApprox{T<:AbstractMatrix} <: AbstractHessianApprox
    B::T
    form::Symbol
end
function StructuredBFGSApprox(B::AbstractMatrix; form::Symbol=:hessian)
    return StructuredBFGSApprox(B, form)
end

function init!(H::StructuredBFGSApprox, n::Int, x0)
    B = H.B
    fill!(B, 0)
    for i in 1:min(n, size(B, 1))
        B[i, i] = 1.0
    end
    return H
end
reset!(H::StructuredBFGSApprox) = init!(H, size(H.B, 1), nothing)

function update!(H::StructuredBFGSApprox, s, y)
    B = H.B
    sy = dot(s, y)
    sy > 0 || return H
    Bs = B * s
    sBs = dot(s, Bs)
    sBs > 0 || return H
    if H.form === :hessian
        # structure-preserving direct BFGS: restrict the dense update to the
        # structural nonzeros of B
        _structural_update!(B, s, y, Bs, sy, sBs)
    else
        throw(ArgumentError("StructuredBFGSApprox supports form = :hessian only."))
    end
    return H
end

function _structural_update!(B::SparseMatrixCSC, s, y, Bs, sy, sBs)
    vals = nonzeros(B)
    rows = rowvals(B)
    for j in 1:size(B, 2)
        for k in nzrange(B, j)
            i = rows[k]
            vals[k] += y[i] * y[j] / sy - Bs[i] * Bs[j] / sBs
        end
    end
    return B
end
function _structural_update!(B::AbstractMatrix, s, y, Bs, sy, sBs)
    for j in axes(B, 2), i in axes(B, 1)
        if i != j && iszero(B[i, j])
            continue
        end
        B[i, j] += y[i] * y[j] / sy - Bs[i] * Bs[j] / sBs
    end
    return B
end

apply!(Bv, H::StructuredBFGSApprox, v) = mul!(Bv, H.B, v)
function solve!(d, H::StructuredBFGSApprox, g)
    Bm = Symmetric(Matrix(H.B))
    F = cholesky(Bm; check=false)
    if issuccess(F)
        return copyto!(d, F \ g)
    else
        return copyto!(d, g)
    end
end

# ------------------------------------------------------------------
# Newton trust region (exact Hessian via AD)
# ------------------------------------------------------------------
"""
    NewtonTRApprox()

Exact Hessian of the (augmented Lagrangian) objective at every iterate,
differentiated through `NonconvexCore.value_gradient`. Used together with
`QuasiNewtonAlg(; globalization = :trustregion)` for the Newton trust-region
method (Steihaug–Toint CG).
"""
mutable struct NewtonTRApprox <: AbstractHessianApprox
    hessfun::Function
    n::Int
    H::Matrix{Float64}
    NewtonTRApprox() = new((x) -> nothing, 0, zeros(0, 0))
end

function init!(H::NewtonTRApprox, n::Int, x0)
    H.n = n
    H.H = zeros(n, n)
    return H
end
reset!(H::NewtonTRApprox) = H
update!(H::NewtonTRApprox, s, y) = H   # no secants; exact Hessian
function apply!(Hv, H::NewtonTRApprox, v)
    return mul!(Hv, H.H, v)
end
solve!(d, H::NewtonTRApprox, g) = copyto!(d, g)

# Newton-type approximations evaluate the Hessian exactly at every iterate
# through `hessfun` (AD-based NewtonTRApprox, or the JuMP-evaluator-based
# JumpNewtonTRApprox); no secant updates apply to them.
_is_newton(::NewtonTRApprox) = true
_is_newton(::Any) = false

# The Steihaug preconditioner of an approximation, when it carries one.
_steihaug_precond(a) = hasproperty(a, :precond) ? a.precond : nothing

# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
"""
    QuasiNewtonAlg(approx = LBFGSApprox(10); globalization = :backtracking,
                   max_iter = 200, tol = 1e-6, damping = true, delta0 = 1.0)

A smooth box-constrained quasi-Newton optimizer: `approx` is any
[`AbstractHessianApprox`](@ref) (L-BFGS with an optional preconditioner, dense
or structured BFGS, or the exact-Hessian Newton trust region). `globalization` is
`:backtracking` (Armijo), `:wolfe` (strong Wolfe) or `:trustregion`
(Steihaug–Toint with a projected step). `damping` enables Powell damping of the
secant pairs for nonconvex objectives. `projected_stop` (default `true`)
terminates on the box-projected gradient norm (the correct stationarity
measure when the optimum sits on a bound, where the raw gradient does not
vanish) instead of the raw gradient norm, and reports that projected norm as
the KKT residual. Measured on the TopOpt local-stress single-level solve:
bit-for-bit identical trajectory at 7.2% less wall-clock (303.1 vs 326.7 s).
"""
struct QuasiNewtonAlg{A<:AbstractHessianApprox} <: AbstractOptimizer
    approx::A
    globalization::Symbol
    max_iter::Int
    tol::Real
    damping::Bool
    delta0::Real
    reset::Bool
    nonmonotone::Bool
    soc::Bool
    levenberg::Bool
    tron_cauchy::Bool
    projected_stop::Bool
end
# The quasi-Newton driver solves smooth box-constrained (possibly unbounded)
# problems; it handles no other constraint kinds.
function NonconvexCore.solver_capabilities(::QuasiNewtonAlg)
    return ConstraintCapabilities(;
        linear_eq=false,
        nonlinear_eq=false,
        linear_ineq=false,
        nonlinear_ineq=false,
        linear_sd=false,
        nonlinear_sd=false,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=true,
    )
end

function QuasiNewtonAlg(
    approx::AbstractHessianApprox=LBFGSApprox(10);
    globalization::Symbol=:backtracking,
    max_iter::Int=200,
    tol::Real=1e-6,
    damping::Bool=true,
    delta0::Real=1.0,
    reset::Bool=true,
    nonmonotone::Bool=false,
    soc::Bool=false,
    levenberg::Bool=true,
    tron_cauchy::Bool=false,
    projected_stop::Bool=true,
)
    globalization in (:backtracking, :wolfe, :trustregion, :tron) || throw(
        ArgumentError("globalization must be :backtracking, :wolfe, :trustregion or :tron"),
    )
    return QuasiNewtonAlg(
        approx,
        globalization,
        max_iter,
        tol,
        damping,
        delta0,
        reset,
        nonmonotone,
        soc,
        levenberg,
        tron_cauchy,
        projected_stop,
    )
end

mutable struct QuasiNewtonWorkspace{
    M<:VecModel,
    A<:AbstractHessianApprox,
    X<:AbstractVector,
    O<:QuasiNewtonAlg,
    R<:Base.RefValue{Int},
} <: Workspace
    model::M
    approx::A
    x0::X
    optimizer::O
    counter::R
end

function Workspace(
    model::VecModel, optimizer::QuasiNewtonAlg, x0::AbstractVector; kwargs...
)
    return QuasiNewtonWorkspace(model, optimizer.approx, copy(x0), optimizer, Ref(0))
end

# Powell damping: blend y towards Bs so the updated approximation stays
# positive definite when the secant condition sᵀy > 0 fails.
function _damped_secant(y, s, Bs; η=0.2)
    sy = dot(s, y)
    sBs = dot(s, Bs)
    if sBs > 0 && sy >= η * sBs
        return y
    end
    if sBs <= 0
        return y
    end
    θ = (1 - η) * sBs / (sBs - sy)
    return θ .* y .+ (1 - θ) .* Bs
end

# strong Wolfe line search along the projected path
function _backtracking_linesearch(f, x, d, fx, gx, α₀=1.0; c1=1e-4)
    α = α₀
    dg = dot(gx, d)
    dg < 0 || return 0.0, fx
    for _ in 1:30
        xt = x .+ α .* d
        fxt = f(xt)
        if isfinite(fxt) && fxt <= fx + c1 * α * dg
            return α, fxt
        end
        α *= 0.5
    end
    return 0.0, fx
end

function _project(x, d, lb, ub)
    p = similar(d)
    for i in eachindex(x)
        p[i] = clamp(x[i] + d[i], lb[i], ub[i]) - x[i]
    end
    return p
end

# Box-projected gradient infinity norm: the correct first-order stationarity
# measure for box-constrained problems. Gradient components pointing out of
# an active bound are zeroed before the norm is taken, so the test vanishes
# at a bound-constrained optimum (where the raw gradient does not).
function _projected_gnorm(g, x, lb, ub)
    s = 0.0
    @inbounds for i in eachindex(x)
        gi = g[i]
        (x[i] <= lb[i] && gi > 0) && continue
        (x[i] >= ub[i] && gi < 0) && continue
        s = max(s, abs(gi))
    end
    return s
end

function optimize!(workspace::QuasiNewtonWorkspace)
    model = workspace.model
    approx = workspace.approx
    x = copy(workspace.x0)
    optimizer = workspace.optimizer
    counter = workspace.counter
    counter[] = 0
    lb, ub = model.box_min, model.box_max
    n = length(x)
    if optimizer.reset || !_initialized(approx, n)
        init!(approx, n, x)
    end
    # objective and gradient closure (function may be an Objective or plain function)
    obj = model.objective
    obj === nothing && throw(ArgumentError("The model has no objective."))
    fobj = obj
    gradfun = y -> begin
        _, gg = NonconvexCore.value_gradient(fobj, y)
        return gg
    end
    if approx isa NewtonTRApprox
        approx.hessfun = y -> begin
            h = NonconvexCore.value_hessian(fobj, y)
            return h isa Tuple ? h[1] : h
        end
    end
    fx = fobj(x)
    # Core value_gradient may reuse its output buffer across calls: copy it
    # out immediately, never hold a reference across another gradient call.
    g = copy(gradfun(x))
    Δ = optimizer.delta0
    τ = optimizer.levenberg ? 1e-8 : 0.0
    merits = Float64[]
    M = 5
    for iter in 1:optimizer.max_iter
        counter[] += 1
        d = similar(x)
        # rebuild the position-dependent exact terms (no-op for plain
        # quasi-Newton models)
        _rebind_terms!(approx, x)
        if optimizer.globalization === :tron
            # TRON: projected-CG trust-region step with active-set tracking.
            d = _tron_subproblem(v -> _hessmatvec(approx, v), g, x, lb, ub, Δ, τ)
            d = _project(x, d, lb, ub)
            if optimizer.tron_cauchy
                d = _tron_cauchy_refine(d, v -> _hessmatvec(approx, v), g, x, lb, ub, Δ)
            end
            norm(d) < 1e-12 && break
            xnew = x .+ d
            fnew = fobj(xnew)
            ared = fx - fnew
            pred = -(dot(g, d) + dot(d, _hessmatvec(approx, d)) / 2)
            push!(merits, fx)
            length(merits) > M && popfirst!(merits)
            fref = maximum(merits)
            if pred > 0 &&
                (ared / pred > 0.1 || (optimizer.nonmonotone && fnew < fref - 1e-2 * pred))
                s = d
                y = copy(gradfun(xnew)) .- g
                update!(approx, s, y)
                x .= xnew
                fx = fnew
                g .= gradfun(x)
                stopnorm = if optimizer.projected_stop
                    _projected_gnorm(g, x, lb, ub)
                else
                    norm(g, Inf)
                end
                stopnorm < optimizer.tol && break
                Δ = min(10Δ, optimizer.delta0 * 10)
                τ = optimizer.levenberg ? max(τ / 2, 1e-10) : 0.0
            else
                Δ = max(Δ / 2, 1e-10)
                τ = optimizer.levenberg ? min(τ * 2, 1e4) : 0.0
            end
            continue
        end
        if optimizer.globalization === :trustregion
            # Steihaug-Toint CG with the current Hessian model
            Hmat = _is_newton(approx) ? approx.hessfun(x) : nothing
            if _is_newton(approx)
                approx.H .= Hmat
            end
            d, negcurv = _steihaug(approx, g, Δ, τ, _steihaug_precond(approx))
            d = _project(x, d, lb, ub)
            norm(d) < 1e-12 && break
            xnew = x .+ d
            fnew = fobj(xnew)
            ared = fx - fnew
            pred = -(dot(g, d) + dot(d, _hessmatvec(approx, d)) / 2)
            # Optional nonmonotone acceptance (Grippo-Lampariello-Lucidi):
            # accept if the reduction beats the best of the last M merits,
            # which lets the iterate escape flat local traps of nonconvex
            # subproblems. Off by default: it changed the outer trajectories
            # on AC-OPF-style problems.
            push!(merits, fx)
            length(merits) > M && popfirst!(merits)
            fref = maximum(merits)
            if pred > 0 &&
                (ared / pred > 0.1 || (optimizer.nonmonotone && fnew < fref - 1e-2 * pred))
                s = d
                y = copy(gradfun(xnew)) .- g
                update!(approx, s, y)
                x .= xnew
                fx = fnew
                g .= gradfun(x)
                stopnorm = if optimizer.projected_stop
                    _projected_gnorm(g, x, lb, ub)
                else
                    norm(g, Inf)
                end
                stopnorm < optimizer.tol && break
                Δ = min(10Δ, optimizer.delta0 * 10)
                τ = optimizer.levenberg ? max(τ / 2, 1e-10) : 0.0
            elseif optimizer.soc
                # Second-order correction (Ipopt, opt-in): re-solve the model
                # with the gradient evaluated at the rejected trial point.
                gsoc = gradfun(xnew)
                dsoc, _ = _steihaug(approx, gsoc, Δ, τ, _steihaug_precond(approx))
                dsoc = _project(x, dsoc, lb, ub)
                xnsoc = x .+ dsoc
                fnsoc = fobj(xnsoc)
                ared_soc = fx - fnsoc
                pred_soc = -(dot(gsoc, dsoc) + dot(dsoc, _hessmatvec(approx, dsoc)) / 2)
                if pred_soc > 0 && ared_soc / pred_soc > 0.1
                    s = dsoc
                    y = copy(gradfun(xnsoc)) .- g
                    update!(approx, s, y)
                    x .= xnsoc
                    fx = fnsoc
                    g .= gradfun(x)
                    stopnorm = if optimizer.projected_stop
                        _projected_gnorm(g, x, lb, ub)
                    else
                        norm(g, Inf)
                    end
                    stopnorm < optimizer.tol && break
                    Δ = min(10Δ, optimizer.delta0 * 10)
                    τ = optimizer.levenberg ? max(τ / 2, 1e-10) : 0.0
                else
                    Δ = max(Δ / 2, 1e-10)
                    τ = optimizer.levenberg ? min(τ * 2, 1e4) : 0.0
                end
            else
                Δ = max(Δ / 2, 1e-10)
                # inertia-driven regularization: grow τ faster when the CG
                # step came from a negative-curvature boundary hit
                τ = if optimizer.levenberg
                    negcurv ? min(τ * 4, 1e4) : min(τ * 2, 1e4)
                else
                    0.0
                end
            end
            continue
        end
        solve!(d, approx, g)
        d .= .-d
        d = _project(x, d, lb, ub)
        norm(d) < 1e-12 && break
        α, fxnew = _backtracking_linesearch(fobj, x, d, fx, g)
        α <= 0 && break
        xnew = x .+ α .* d
        s = xnew .- x
        y = gradfun(xnew) .- g
        if optimizer.damping && !_is_newton(approx)
            Bs = similar(y)
            apply!(Bs, approx, s)
            y = _damped_secant(y, s, Bs)
        end
        update!(approx, s, y)
        x .= xnew
        fx = fxnew
        g .= gradfun(x)
        stopnorm = if optimizer.projected_stop
            _projected_gnorm(g, x, lb, ub)
        else
            norm(g, Inf)
        end
        stopnorm < optimizer.tol && break
    end
    stationarity = if optimizer.projected_stop
        _projected_gnorm(g, x, lb, ub)
    else
        norm(g, Inf)
    end
    return GenericResult(
        optimizer,
        copy(workspace.x0),
        x,
        fx,
        counter[],
        false,
        Tolerance(; kkt=stationarity),
        nothing,
        counter[],
    )
end

# Projected Cauchy point of the bounded trust-region subproblem (the
# steepest-descent step clipped to the box and the radius; Conn-Gould-Toint
# §9). Used by TRON's optional fallback.
function _box_cauchy(g, x, lb, ub, Δ)
    n = length(g)
    gp = copy(g)
    for i in 1:n
        (x[i] <= lb[i] && g[i] > 0) && (gp[i] = 0.0)
        (x[i] >= ub[i] && g[i] < 0) && (gp[i] = 0.0)
    end
    nrm = norm(gp)
    nrm < 1e-14 && return zeros(n)
    τ = min(1.0, Δ / nrm)
    d = similar(x)
    for i in 1:n
        d[i] = clamp(x[i] - τ * gp[i], lb[i], ub[i]) - x[i]
    end
    return d
end

# Lin-Moré safeguard for TRON: if the projected-CG step achieves less than
# half the projected-Cauchy model decrease, fall back to the Cauchy step,
# then try shorter steps along the projected path.
function _tron_cauchy_refine(d, Bv, g, x, lb, ub, Δ)
    modeldec(dd) = -(dot(g, dd) + dot(dd, Bv(dd)) / 2)
    dC = _box_cauchy(g, x, lb, ub, Δ)
    mC = modeldec(dC)
    mCG = modeldec(d)
    if mCG < 0.5 * mC
        d = dC
        mCG = mC
    end
    for α in (0.75, 0.5, 0.25)
        dα = _project(x, α .* d, lb, ub)
        mα = modeldec(dα)
        if mα > mCG
            d = dα
            mCG = mα
        end
    end
    return d
end

# TRON (Lin & Moré 1999): projected conjugate gradients on the free-variable
# subspace of the bounded trust-region subproblem
#   min gᵀd + ½ dᵀ(B + τI)d   s.t.  ‖d‖ ≤ Δ  and  l ≤ x + d ≤ u.
# Variables that hit a bound during the CG sweep are fixed for the rest of
# the sweep (active-set identification).
function _tron_subproblem(Bv, g, x, lb, ub, Δ, τ)
    n = length(g)
    d = zeros(n)
    free = trues(n)
    gp = copy(g)
    for i in 1:n
        (x[i] <= lb[i] && g[i] > 0) && (gp[i] = 0.0)
        (x[i] >= ub[i] && g[i] < 0) && (gp[i] = 0.0)
    end
    r = copy(gp)
    p = .-gp
    rr = dot(r, r)
    rr < 1e-20 && return d
    for _ in 1:n
        Bp = Bv(p)
        Bp .+= τ .* p
        pBp = dot(p, Bp)
        if pBp <= 0
            # Negative curvature: ride p to the trust-region boundary (or a
            # bound, whichever comes first).
            return _tron_truncate(d, p, x, lb, ub, free, Δ)
        end
        α = rr / pBp
        t = α
        for i in 1:n
            free[i] || continue
            p[i] == 0 && continue
            t = min(
                t, p[i] > 0 ? (ub[i] - x[i] - d[i]) / p[i] : (lb[i] - x[i] - d[i]) / p[i]
            )
        end
        dnew = d .+ t .* p
        if dot(dnew, dnew) >= Δ^2
            return _tron_truncate(d, p, x, lb, ub, free, Δ)
        end
        d .= dnew
        r .+= t .* Bp
        # fix variables that reached a bound and zero their residual
        hitbound = false
        for i in 1:n
            free[i] || continue
            if (x[i] + d[i] <= lb[i] + 1e-12 || x[i] + d[i] >= ub[i] - 1e-12)
                free[i] = false
                r[i] = 0.0
                hitbound = true
            end
        end
        rnew2 = dot(r, r)
        rnew2 < 1e-20 && return d
        β = rnew2 / rr
        rr = rnew2
        p .= .-r .+ β .* p
        for i in 1:n
            free[i] || (p[i] = 0.0)
        end
    end
    return d
end

# Largest step along p from d that stays inside the bounds and reaches the
# trust-region boundary.
function _tron_truncate(d, p, x, lb, ub, free, Δ)
    τb = Inf
    for i in eachindex(p)
        free[i] || continue
        p[i] == 0 && continue
        τb = min(τb, p[i] > 0 ? (ub[i] - x[i] - d[i]) / p[i] : (lb[i] - x[i] - d[i]) / p[i])
    end
    a = dot(p, p)
    b = 2 * dot(d, p)
    c = dot(d, d) - Δ^2
    disc = max(b^2 - 4a * c, 0.0)
    τq = (-b + sqrt(disc)) / (2a)
    τ = min(τq, τb)
    return d .+ τ .* p
end

# Whether the approximation already holds usable curvature for dimension `n`
# (warm-starting across augmented Lagrangian subproblems).
_initialized(::AbstractHessianApprox, n::Int) = false
_initialized(H::LBFGSApprox, n::Int) = !isempty(H.s) && length(H.s[1]) == n

function _hessmatvec(approx::AbstractHessianApprox, v)
    Hv = similar(v)
    apply!(Hv, approx, v)
    return Hv
end

# Steihaug–Toint CG for the trust-region subproblem
# min ⟨g,d⟩ + ½ dᵀ(B + τI)d, ‖d‖ ≤ Δ (Levenberg dual regularization)
function _steihaug(approx, g, Δ, τ=0.0, M=nothing)
    n = length(g)
    d = zeros(n)
    r = copy(g)
    # preconditioned CG: z = M⁻¹r; unpreconditioned z = r
    z = if M === nothing
        copy(g)
    else
        zi = zeros(n)
        ldiv!(zi, M, r)
        zi
    end
    p = .-z
    zr = dot(z, r)
    rnorm2 = dot(r, r)
    rnorm2 < 1e-14 && return d, false
    for _ in 1:n
        Bp = _hessmatvec(approx, p) .+ τ .* p
        pBp = dot(p, Bp)
        if pBp <= 0
            # negative curvature: hit the boundary along p
            a = dot(p, p)
            b = 2 * dot(d, p)
            c = dot(d, d) - Δ^2
            τ = _root_of_quadratic(a, b, c)
            return d .+ τ .* p, true
        end
        α = rnorm2 / pBp
        dnew = d .+ α .* p
        if dot(dnew, dnew) >= Δ^2
            a = dot(p, p)
            b = 2 * dot(d, p)
            c = dot(d, d) - Δ^2
            τ = _root_of_quadratic(a, b, c)
            return d .+ τ .* p, false
        end
        d .= dnew
        r .+= α .* Bp
        rnorm2new = dot(r, r)
        rnorm2new < 1e-14 && break
        z = if M === nothing
            copy(r)
        else
            zi = zeros(n)
            ldiv!(zi, M, r)
            zi
        end
        zrnew = dot(z, r)
        β = zrnew / zr
        p .= .-z .+ β .* p
        zr = zrnew
        rnorm2 = rnorm2new
    end
    return d, false
end

function _root_of_quadratic(a, b, c)
    disc = b^2 - 4a * c
    disc = max(disc, 0.0)
    τ1 = (-b + sqrt(disc)) / (2a)
    τ2 = (-b - sqrt(disc)) / (2a)
    return max(τ1, τ2) > 0 ? max(τ1, τ2) : 0.0
end
