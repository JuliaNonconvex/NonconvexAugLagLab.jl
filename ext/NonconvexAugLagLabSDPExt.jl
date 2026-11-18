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
using NonconvexSemidefinite: NonconvexSemidefinite
using NonconvexSemidefinite: project_psd, psd_infeasibility
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

_logdet_term(Y, ρ, B) = if isposdef(Symmetric(B))
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

end
