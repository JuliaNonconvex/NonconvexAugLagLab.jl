# Semidefinite relaxation for AugLagAlg.
#
# Projected augmented Lagrangian (Sun-Sun-Zhang, the default): the penalty
# term is
#     φ(x) = (1/2ρ) (‖Π_{S₊ⁿ}(Y - ρ P(x))‖²_F - ‖Y‖²_F)
# with multiplier update Y ← Π_{S₊ⁿ}(Y - ρ P(x)). Gradients flow through the
# analytic rrule of `project_psd`.
#
# Logdet nonlinear rescaling (`penalty = :logdet` / `:mixed_logdet`,
# Polyak-Teboulle / Kočvara-Stingl modified barrier for the PSD cone,
# clean-room): the penalty term is
#     φ(x) = -(1/ρ) logdet P(x) + ⟨Y, P(x)⟩     (defined for P ≻ 0)
# with multiplier update Y ← Π_{S₊ⁿ}(P(x)⁻¹/ρ) (the barrier-stationarity
# dual estimate, safeguarded). Outside the cone the projected quadratic
# term takes over, so the subproblems stay defined everywhere.
module NonconvexAugLagLabSDPExt

using NonconvexAugLagLab
using NonconvexCore: VecModel
import NonconvexAugLagLab: _sd_violation, _sd_terms, _sd_multiplier_update!, _sd_init
import NonconvexAugLagLab:
    _set_psd_violation, _set_psd_init, _set_psd_term, _set_psd_update!
import NonconvexAugLagLab: _set_is_psd, _component_dim
using NonconvexSemidefinite: NonconvexSemidefinite
using NonconvexSemidefinite: project_psd, psd_infeasibility, PSDCone
using NonconvexSemidefinite: NonconvexSemidefinite, ChainRulesCore
# Block-aware helpers when a NonconvexSemidefinite with BlockSDConstraint is
# loaded; fall back to the single-matrix contract for the registered 0.3.x.
if isdefined(NonconvexSemidefinite, :_sd_blocks)
    using NonconvexSemidefinite: _sd_blocks, _sd_dims
else
    _sd_blocks(c::NonconvexCore.SDConstraint, x) = (c(x),)
    _sd_dims(c::NonconvexCore.SDConstraint) = [c.dim]
end
using LinearAlgebra

# The PSDCone components of function-in-set constraints are semidefinite.
NonconvexAugLagLab._set_is_psd(::PSDCone) = true
NonconvexAugLagLab._component_dim(s::PSDCone) = s.dim

# Zygote cannot trace through LAPACK (`logdet` -> foreigncall); the analytic
# rrule d/dP logdet(Symmetric(P)) = inv(P)' feeds the quasi-Newton gradient.
_sd_logdet(P::AbstractMatrix) = logdet(Symmetric(P))
function ChainRulesCore.rrule(::typeof(_sd_logdet), P::AbstractMatrix)
    y = logdet(Symmetric(P))
    return y, Δ -> (ChainRulesCore.NoTangent(), Δ .* inv(Matrix(Symmetric(P)))')
end

function NonconvexAugLagLab._sd_violation(model::VecModel, i::Int, x::AbstractVector)
    return [psd_infeasibility(B) for B in _sd_blocks(model.sd_constraints.fs[i], x)]
end

function NonconvexAugLagLab._sd_init(model::VecModel, sd_rel, ρ)
    return [zeros(d, d) for sd in model.sd_constraints.fs[sd_rel] for d in _sd_dims(sd)]
end

_projected_term(Y, ρ, B) = begin
    Z = project_psd(Y - ρ * B)
    (sum(abs2, Z) - sum(abs2, Y)) / (2ρ)
end

_logdet_term(Y, ρ, B) =
    if isposdef(Symmetric(B))
        -(1 / ρ) * _sd_logdet(B) + dot(Y, B)
    else
        _projected_term(Y, ρ, B)
    end

function NonconvexAugLagLab._sd_terms(model::VecModel, sd_rel, ρ, x, Y, penalty)
    fns = model.sd_constraints.fs
    use_logdet = penalty in (:logdet, :mixed_logdet)
    return x -> begin
        val = 0.0
        k = 1
        for i in sd_rel
            for B in _sd_blocks(fns[i], x)
                val += use_logdet ? _logdet_term(Y[k], ρ, B) : _projected_term(Y[k], ρ, B)
                k += 1
            end
        end
        return val
    end
end

function NonconvexAugLagLab._sd_multiplier_update!(
    model::VecModel, sd_rel, ρ, x, Y, penalty
)
    use_logdet = penalty in (:logdet, :mixed_logdet)
    k = 1
    for i in sd_rel
        for B in _sd_blocks(model.sd_constraints.fs[i], x)
            Y[k] = if use_logdet && isposdef(Symmetric(B))
                # Barrier-AL step (experimental): U ← Π(P⁻¹/ρ). Note: at the
                # fixed point the linear term ⟨U,P⟩ cancels the barrier
                # gradient exactly (the subproblem reduces to ∇f = 0), which
                # is why the kernel lands short of the projected quadratic
                # on the covariance MLE; a proper cone-PBM needs a
                # growing-weight scalar barrier. Kept experimental.
                project_psd(inv(Symmetric(B)) / ρ)
            else
                project_psd(Y[k] - ρ * B)
            end
            k += 1
        end
    end
    return Y
end

# ------------------------------------------------------------------
# PSD components of function-in-set constraints: the same projected ALM,
# applied to the single matrix each PSDCone component evaluates to (the
# chordal transform flattens products into one PSDCone per clique).
# ------------------------------------------------------------------

function NonconvexAugLagLab._set_psd_violation(vi::AbstractMatrix, sc)
    return [psd_infeasibility(vi) / sc]
end

function NonconvexAugLagLab._set_psd_init(c::FunctionInSet, ρ)
    comps = NonconvexCore.components(c.set)
    return [zeros(d, d) for s in comps if s isa PSDCone for d in (s.dim,)]
end

function NonconvexAugLagLab._set_psd_term(Y::AbstractMatrix, ρ, vi::AbstractMatrix, penalty)
    use_logdet = penalty in (:logdet, :mixed_logdet)
    return use_logdet ? _logdet_term(Y, ρ, vi) : _projected_term(Y, ρ, vi)
end

# ------------------------------------------------------------------
# Batched PSD penalty: ONE call with an analytic rrule instead of a
# per-block loop, so reverse-mode AD sees a single primitive. The PSD
# components are selected from the raw bundle value/components inside the
# native function — no traced filtering or tuple construction.
# ------------------------------------------------------------------

"""
    _set_psd_terms(Y, ρ, v::Tuple, comps::Tuple, penalty::Symbol)

The combined penalty of all semidefinite components of one bundle
evaluation `v` (the `PSDCone` components of `comps`, with one multiplier
matrix per component in `Y`): the sum of the per-block projected or logdet
terms. Defined with an analytic [`rrule`](@ref): the selection and the
block loop are native in the forward pass, and the pullback scatters the
cotangent back to the PSD slots of `v` (all other slots get `NoTangent`).

The projected branch has ∂φ/∂P = -Π₊(Y - ρP) and ∂φ/∂Y = (Π₊(Y - ρP) - Y)/ρ;
the logdet branch (P ≻ 0) has ∂φ/∂P = Y - P⁻¹/ρ and ∂φ/∂Y = P.
"""
function NonconvexAugLagLab._set_psd_terms(
    Y::AbstractVector, ρ, v::Tuple, comps::Tuple, scales, penalty::Symbol
)
    use_logdet = penalty in (:logdet, :mixed_logdet)
    s = 0.0
    k = 1
    for (idx, (vi, set)) in enumerate(zip(v, comps))
        set isa PSDCone || continue
        P = vi ./ scales[idx]
        s += use_logdet ? _logdet_term(Y[k], ρ, P) : _projected_term(Y[k], ρ, P)
        k += 1
    end
    return s
end

function ChainRulesCore.rrule(
    ::typeof(NonconvexAugLagLab._set_psd_terms),
    Y::AbstractVector,
    ρ,
    v::Tuple,
    comps::Tuple,
    scales,
    penalty::Symbol,
)
    use_logdet = penalty in (:logdet, :mixed_logdet)
    s = 0.0
    # per-block reverse records: the projected matrix (or the inverse for
    # the logdet branch) and the branch flag; native references only.
    Zs = Matrix{Float64}[]
    logdet_flags = Bool[]
    k = 1
    for (idx, (vi, set)) in enumerate(zip(v, comps))
        set isa PSDCone || continue
        P = vi ./ scales[idx]
        if use_logdet && isposdef(Symmetric(P))
            push!(logdet_flags, true)
            push!(Zs, inv(Matrix(Symmetric(P))))
            s += -(1 / ρ) * _sd_logdet(P) + dot(Y[k], P)
        else
            Zk = project_psd(Y[k] - ρ * P)
            push!(logdet_flags, false)
            push!(Zs, Zk)
            s += (sum(abs2, Zk) - sum(abs2, Y[k])) / (2ρ)
        end
        k += 1
    end
    function _set_psd_terms_pullback(Δs)
        Δs = ChainRulesCore.unthunk(Δs)
        if Δs isa ChainRulesCore.AbstractZero
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ntuple(_ -> ChainRulesCore.NoTangent(), length(v)),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end
        Δv = Vector{Any}(undef, length(v))
        ΔYs = Vector{Matrix{Float64}}(undef, length(Zs))
        Δρ = 0.0
        k = 1
        for (idx, (vi, set)) in enumerate(zip(v, comps))
            if set isa PSDCone
                sc = scales[idx]
                P = vi ./ sc
                Yk = Y[k]
                if logdet_flags[k]
                    Binv = Zs[k]
                    # φ = -(1/ρ) logdet P + ⟨Y, P⟩, P = vi/sc
                    Δv[idx] = Δs .* ((Yk .- Binv ./ ρ) ./ sc)
                    ΔYs[k] = Δs .* P
                    Δρ += Δs * (_sd_logdet(P) / ρ^2)
                else
                    Zk = Zs[k]
                    # φ = (‖Π₊(Y - ρP)‖² - ‖Y‖²) / (2ρ), P = vi/sc
                    Δv[idx] = Δs .* (-Zk ./ sc)
                    ΔYs[k] = Δs .* ((Zk .- Yk) ./ ρ)
                    Δρ += Δs * (-(sum(abs2, Zk) - sum(abs2, Yk)) / (2ρ^2) - dot(Zk, P) / ρ)
                end
                k += 1
            else
                Δv[idx] = ChainRulesCore.NoTangent()
            end
        end
        return (
            ChainRulesCore.NoTangent(),
            ΔYs,
            Δρ,
            Tuple(Δv),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end
    return s, _set_psd_terms_pullback
end

function NonconvexAugLagLab._set_psd_update!(
    Y::AbstractMatrix, ρ, vi::AbstractMatrix, penalty
)
    Y .= project_psd(Y - ρ * vi)
    return nothing
end

# ------------------------------------------------------------------
# The multiplier terms Σₖ⟨Yₖ, Pₖ⟩ of the PSD components of one bundle —
# linear in both arguments, batched into ONE call with an rrule for the
# KKT-complete stopping test.
# ------------------------------------------------------------------

function NonconvexAugLagLab._set_psd_multiplier_terms(
    Y::AbstractVector, v::Tuple, comps::Tuple, scales
)
    s = 0.0
    k = 1
    for (idx, (vi, set)) in enumerate(zip(v, comps))
        set isa PSDCone || continue
        s += dot(Y[k], vi ./ scales[idx])
        k += 1
    end
    return s
end

function ChainRulesCore.rrule(
    ::typeof(NonconvexAugLagLab._set_psd_multiplier_terms),
    Y::AbstractVector,
    v::Tuple,
    comps::Tuple,
    scales,
)
    s = 0.0
    k = 1
    for (idx, (vi, set)) in enumerate(zip(v, comps))
        set isa PSDCone || continue
        s += dot(Y[k], vi ./ scales[idx])
        k += 1
    end
    function _set_psd_multiplier_terms_pullback(Δs)
        Δs = ChainRulesCore.unthunk(Δs)
        if Δs isa ChainRulesCore.AbstractZero
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ntuple(_ -> ChainRulesCore.NoTangent(), length(v)),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end
        Δv = Vector{Any}(undef, length(v))
        ΔYs = Vector{Matrix{Float64}}(undef, length(Y))
        k = 1
        for (idx, (vi, set)) in enumerate(zip(v, comps))
            if set isa PSDCone
                Δv[idx] = Δs .* (Y[k] ./ scales[idx])
                ΔYs[k] = Δs .* (vi ./ scales[idx])
                k += 1
            else
                Δv[idx] = ChainRulesCore.NoTangent()
            end
        end
        return (ChainRulesCore.NoTangent(), ΔYs, Tuple(Δv), ChainRulesCore.NoTangent())
    end
    return s, _set_psd_multiplier_terms_pullback
end

end
