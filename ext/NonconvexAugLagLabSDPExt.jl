# Semidefinite relaxation for AugLagAlg via the projected augmented Lagrangian
# (Sun-Sun-Zhang): the penalty term is
#     φ(x) = (1/2ρ) (‖Π_{S₊ⁿ}(Y - ρ P(x))‖²_F - ‖Y‖²_F)
# with multiplier update Y ← Π_{S₊ⁿ}(Y - ρ P(x)). Gradients flow through the
# analytic rrule of `project_psd`.
module NonconvexAugLagLabSDPExt

using NonconvexAugLagLab
using NonconvexCore: VecModel
import NonconvexAugLagLab: _sd_violation, _sd_terms, _sd_multiplier_update!, _sd_init
using NonconvexSemidefinite: NonconvexSemidefinite
using NonconvexSemidefinite: project_psd, psd_infeasibility
# Block-aware helpers when a NonconvexSemidefinite with BlockSDConstraint is
# loaded; fall back to the single-matrix contract for the registered 0.3.x.
if isdefined(NonconvexSemidefinite, :_sd_blocks)
    using NonconvexSemidefinite: _sd_blocks, _sd_dims
else
    _sd_blocks(c::NonconvexCore.SDConstraint, x) = (c(x),)
    _sd_dims(c::NonconvexCore.SDConstraint) = [c.dim]
end
using LinearAlgebra

function NonconvexAugLagLab._sd_violation(model::VecModel, i::Int, x::AbstractVector)
    return [psd_infeasibility(B) for B in _sd_blocks(model.sd_constraints.fs[i], x)]
end

function NonconvexAugLagLab._sd_init(model::VecModel, sd_rel, ρ)
    return [zeros(d, d) for sd in model.sd_constraints.fs[sd_rel] for d in _sd_dims(sd)]
end

function NonconvexAugLagLab._sd_terms(model::VecModel, sd_rel, ρ, x, Y)
    fns = model.sd_constraints.fs
    return x -> begin
        val = 0.0
        k = 1
        for i in sd_rel
            for B in _sd_blocks(fns[i], x)
                Z = project_psd(Y[k] - ρ * B)
                val += (sum(abs2, Z) - sum(abs2, Y[k])) / (2ρ)
                k += 1
            end
        end
        return val
    end
end

function NonconvexAugLagLab._sd_multiplier_update!(model::VecModel, sd_rel, ρ, x, Y)
    k = 1
    for i in sd_rel
        for B in _sd_blocks(model.sd_constraints.fs[i], x)
            Y[k] = project_psd(Y[k] - ρ * B)
            k += 1
        end
    end
    return Y
end

end
