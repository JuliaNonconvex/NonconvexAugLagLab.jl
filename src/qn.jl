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
        # factorization-free robust solve: Cholesky of the SPD approximation
        F = cholesky(Symmetric(H.H); check=false)
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
function StructuredBFGSApprox(B::AbstractMatrix; form::Symbol=:hessian)
    return StructuredBFGSApprox(B, form)
end

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
    reset::Bool
    nonmonotone::Bool
    soc::Bool
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
    linesearch::Symbol=:backtracking,
    max_iter::Int=200,
    tol::Real=1e-6,
    damping::Bool=true,
    Δ0::Real=1.0,
    reset::Bool=true,
    nonmonotone::Bool=false,
    soc::Bool=false,
)
    linesearch in (:backtracking, :wolfe, :trustregion, :tron) || throw(
        ArgumentError("linesearch must be :backtracking, :wolfe, :trustregion or :tron")
    )
    return QuasiNewtonAlg(
        approx, linesearch, max_iter, tol, damping, Δ0, reset, nonmonotone, soc
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
    Δ = optimizer.Δ0
    τ = 1e-8
    merits = Float64[]
    M = 5
    for iter in 1:optimizer.max_iter
        counter[] += 1
        d = similar(x)
        if optimizer.linesearch === :tron
            # TRON: projected-CG trust-region step with active-set tracking.
            d = _tron_subproblem(v -> _hessmatvec(approx, v), g, x, lb, ub, Δ, τ)
            d = _project(x, d, lb, ub)
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
                norm(g, Inf) < optimizer.tol && break
                Δ = min(10Δ, optimizer.Δ0 * 10)
                τ = max(τ / 2, 1e-10)
            else
                Δ = max(Δ / 2, 1e-10)
                τ = min(τ * 2, 1e4)
            end
            continue
        end
        if optimizer.linesearch === :trustregion
            # Steihaug-Toint CG with the current Hessian model
            Hmat = approx isa NewtonTRApprox ? approx.hessfun(x) : nothing
            if approx isa NewtonTRApprox
                approx.H .= Hmat
            end
            d = _steihaug(approx, g, Δ, τ)
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
                norm(g, Inf) < optimizer.tol && break
                Δ = min(10Δ, optimizer.Δ0 * 10)
                τ = max(τ / 2, 1e-10)
            elseif optimizer.soc
                # Second-order correction (Ipopt, opt-in): re-solve the model
                # with the gradient evaluated at the rejected trial point.
                gsoc = gradfun(xnew)
                dsoc = _steihaug(approx, gsoc, Δ, τ)
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
                    norm(g, Inf) < optimizer.tol && break
                    Δ = min(10Δ, optimizer.Δ0 * 10)
                    τ = max(τ / 2, 1e-10)
                else
                    Δ = max(Δ / 2, 1e-10)
                    τ = min(τ * 2, 1e4)
                end
            else
                Δ = max(Δ / 2, 1e-10)
                τ = min(τ * 2, 1e4)
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
        Tolerance(; kkt=optimizer.tol),
        nothing,
        counter[],
    )
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
function _steihaug(approx, g, Δ, τ=0.0)
    n = length(g)
    d = zeros(n)
    r = copy(g)
    p = .-g
    rnorm2 = dot(r, r)
    rnorm2 < 1e-14 && return d
    for _ in 1:n
        Bp = _hessmatvec(approx, p) .+ τ .* p
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
    return if max(τ1, τ2) > 0
        max(τ1, τ2)
    elseif min(τ1, τ2) > 0
        min(τ1, τ2)
    else
        0.0
    end
end
