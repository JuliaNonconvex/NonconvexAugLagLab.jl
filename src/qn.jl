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
function LBFGSApprox(memory::Int = 10; H0 = IdentityPreconditioner())
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
    m == 1 && println("  solve! state: s=", H.s, " ρ=", H.ρ, " g=", g)
    q = copy(g)
    for i = m:-1:1
        H.α[i] = H.ρ[i] * dot(H.s[i], q)
        q .-= H.α[i] .* H.y[i]
    end
    solve!(d, H.H0, q)
    for i = 1:m
        β = H.ρ[i] * dot(H.y[i], d)
        d .+= (H.α[i] - β) .* H.s[i]
    end
    return d
end

# Apply the direct approximation: B v (two-loop recursion with the H0 operator).
function apply!(Bv, H::LBFGSApprox, v)
    m = length(H.s)
    resize!(H.α, m)
    q = copy(v)
    for i = m:-1:1
        H.α[i] = H.ρ[i] * dot(H.s[i], q)
        q .-= H.α[i] .* H.y[i]
    end
    apply!(Bv, H.H0, q)
    for i = 1:m
        β = H.ρ[i] * dot(H.y[i], Bv)
        Bv .+= (H.α[i] - β) .* H.s[i]
    end
    return Bv
end

function update!(H::LBFGSApprox, s, y)
    sy = dot(s, y)
    length(H.s) < 2 && println("  update!: s=", s, " y=", y, " sy=", sy)
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
DenseBFGSApprox(; form::Symbol = :inverse) = DenseBFGSApprox(zeros(0, 0), form)

function init!(H::DenseBFGSApprox, n::Int, x0)
    H.H = Matrix{Float64}(I, n, n)
    return H
end
reset!(H::DenseBFGSApprox) = (H.H = Matrix{Float64}(I, size(H.H, 1), size(H.H, 1)); H)

function update!(H::DenseBFGSApprox, s, y)
    B = H.H
    sy = dot(s, y)
    sy > 0 || return H
    if H.form === :inverse
        # inverse BFGS update
        ρ = inv(sy)
        Bs = B * s
        H.H .= (I - ρ .* s .* y') * B * (I - ρ .* y .* s') .+ ρ .* s .* s'
    else
        Bs = B * s
        sBs = dot(s, Bs)
        H.H .= B .+ (y .* y') ./ sy .- (Bs .* Bs') ./ sBs
    end
    return H
end

apply!(Bv, H::DenseBFGSApprox, v) = mul!(Bv, H.H, v)

function solve!(d, H::DenseBFGSApprox, g)
    if H.form === :inverse
        return mul!(d, H.H, g)
    else
        # factorization-free robust solve: Cholesky of the SPD approximation
        F = cholesky(Symmetric(H.H), check = false)
        if issuccess(F)
            return copyto!(d, F \ g)
        else
            return copyto!(d, g)   # fall back to steepest descent
        end
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
StructuredBFGSApprox(B::AbstractMatrix; form::Symbol = :hessian) =
    StructuredBFGSApprox(B, form)

function init!(H::StructuredBFGSApprox, n::Int, x0)
    H.B .= 0
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
    F = cholesky(Bm, check = false)
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
`QuasiNewtonAlg(; linesearch = :trustregion)` for the Newton trust-region
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

# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
"""
    QuasiNewtonAlg(approx = LBFGSApprox(10); linesearch = :backtracking,
                   max_iter = 200, tol = 1e-6, damping = true, Δ0 = 1.0)

A smooth box-constrained quasi-Newton optimizer: `approx` is any
[`AbstractHessianApprox`](@ref) (L-BFGS with an optional preconditioner, dense
or structured BFGS, or the exact-Hessian Newton trust region). `linesearch` is
`:backtracking` (Armijo), `:wolfe` (strong Wolfe) or `:trustregion`
(Steihaug–Toint with a projected step). `damping` enables Powell damping of the
secant pairs for nonconvex objectives.
"""
struct QuasiNewtonAlg{A<:AbstractHessianApprox} <: AbstractOptimizer
    approx::A
    linesearch::Symbol
    max_iter::Int
    tol::Real
    damping::Bool
    Δ0::Real
end
# The quasi-Newton driver solves smooth box-constrained (possibly unbounded)
# problems; it handles no other constraint kinds.
NonconvexCore.solver_capabilities(::QuasiNewtonAlg) = ConstraintCapabilities(
    eq = false,
    ineq = false,
    sd = false,
    box = true,
    integer = false,
    infinite_bounds = true,
)

function QuasiNewtonAlg(
    approx::AbstractHessianApprox = LBFGSApprox(10);
    linesearch::Symbol = :backtracking,
    max_iter::Int = 200,
    tol::Real = 1e-6,
    damping::Bool = true,
    Δ0::Real = 1.0,
)
    linesearch in (:backtracking, :wolfe, :trustregion) ||
        throw(ArgumentError("linesearch must be :backtracking, :wolfe or :trustregion"))
    return QuasiNewtonAlg(approx, linesearch, max_iter, tol, damping, Δ0)
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

function Workspace(model::VecModel, optimizer::QuasiNewtonAlg, x0::AbstractVector; kwargs...)
    return QuasiNewtonWorkspace(model, optimizer.approx, copy(x0), optimizer, Ref(0))
end

# Powell damping: blend y towards Bs so the updated approximation stays
# positive definite when the secant condition sᵀy > 0 fails.
function _damped_secant(y, s, Bs; η = 0.2)
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
function _backtracking_linesearch(f, x, d, fx, gx, α₀ = 1.0; c1 = 1e-4)
    α = α₀
    dg = dot(gx, d)
    dg < 0 || return 0.0, fx
    for _ = 1:30
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

function optimize!(workspace::QuasiNewtonWorkspace)
    model = workspace.model
    approx = workspace.approx
    x = copy(workspace.x0)
    optimizer = workspace.optimizer
    counter = workspace.counter
    counter[] = 0
    lb, ub = model.box_min, model.box_max
    n = length(x)
    init!(approx, n, x)
    # objective and gradient closure (function may be an Objective or plain function)
    fobj = model.objective
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
    Δ = optimizer.Δ0
    for iter = 1:optimizer.max_iter
        counter[] += 1
        d = similar(x)
        if optimizer.linesearch === :trustregion
            # Steihaug-Toint CG with the current Hessian model
            Hmat = approx isa NewtonTRApprox ? approx.hessfun(x) : nothing
            if approx isa NewtonTRApprox
                approx.H .= Hmat
            end
            d = _steihaug(approx, g, Δ)
            d = _project(x, d, lb, ub)
            norm(d) < 1e-12 && break
            xnew = x .+ d
            fnew = fobj(xnew)
            ared = fx - fnew
            pred = -(dot(g, d) + dot(d, _hessmatvec(approx, d)) / 2)
            if pred > 0 && ared / pred > 0.1
                s = d
                y = copy(gradfun(xnew)) .- g
                update!(approx, s, y)
                x .= xnew
                fx = fnew
                g .= gradfun(x)
                Δ = min(10Δ, optimizer.Δ0 * 10)
            else
                Δ = max(Δ / 2, 1e-10)
            end
            continue
        end
        solve!(d, approx, g)
        iter == 1 && println("  chk1 g=", g)
        d .= .-d
        iter == 1 && println("  chk2 g=", g)
        d = _project(x, d, lb, ub)
        iter == 1 && println("  chk3 g=", g)
        norm(d) < 1e-12 && break
        α, fxnew = _backtracking_linesearch(fobj, x, d, fx, g)
        iter == 1 && println("  chk4 g=", g, " α=", α)
        α <= 0 && break
        xnew = x .+ α .* d
        s = xnew .- x
        iter == 1 && println("  chk5 g=", g)
        y = gradfun(xnew) .- g
        iter == 1 && println("  chk6 g=", g, " y=", y)
        if optimizer.damping && !(approx isa NewtonTRApprox)
            Bs = similar(y)
            apply!(Bs, approx, s)
            y = _damped_secant(y, s, Bs)
        end
        update!(approx, s, y)
        x .= xnew
        fx = fxnew
        g .= gradfun(x)
        norm(g, Inf) < optimizer.tol && break
    end
    return GenericResult(
        optimizer,
        copy(workspace.x0),
        x,
        fx,
        counter[],
        false,
        Tolerance(kkt = optimizer.tol),
        nothing,
        counter[],
    )
end

function _hessmatvec(approx::AbstractHessianApprox, v)
    Hv = similar(v)
    apply!(Hv, approx, v)
    return Hv
end

# Steihaug–Toint CG for the trust-region subproblem min ⟨g,d⟩ + ½ dᵀBd, ‖d‖ ≤ Δ
function _steihaug(approx, g, Δ)
    n = length(g)
    d = zeros(n)
    r = copy(g)
    p = .-g
    rnorm2 = dot(r, r)
    rnorm2 < 1e-14 && return d
    for _ = 1:n
        Bp = _hessmatvec(approx, p)
        pBp = dot(p, Bp)
        if pBp <= 0
            # negative curvature: hit the boundary along p
            a = dot(p, p)
            b = 2 * dot(d, p)
            c = dot(d, d) - Δ^2
            τ = _root_of_quadratic(a, b, c)
            return d .+ τ .* p
        end
        α = rnorm2 / pBp
        dnew = d .+ α .* p
        if dot(dnew, dnew) >= Δ^2
            a = dot(p, p)
            b = 2 * dot(d, p)
            c = dot(d, d) - Δ^2
            τ = _root_of_quadratic(a, b, c)
            return d .+ τ .* p
        end
        d .= dnew
        r .+= α .* Bp
        rnorm2new = dot(r, r)
        rnorm2new < 1e-14 && break
        β = rnorm2new / rnorm2
        p .= .-r .+ β .* p
        rnorm2 = rnorm2new
    end
    return d
end

function _root_of_quadratic(a, b, c)
    disc = b^2 - 4a * c
    disc = max(disc, 0.0)
    τ1 = (-b + sqrt(disc)) / (2a)
    τ2 = (-b - sqrt(disc)) / (2a)
    return max(τ1, τ2) > 0 ? max(τ1, τ2) : min(τ1, τ2) > 0 ? min(τ1, τ2) : 0.0
end
