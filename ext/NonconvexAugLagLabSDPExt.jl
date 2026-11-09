# Semidefinite relaxation for AugLagAlg via the projected augmented Lagrangian
# (Sun-Sun-Zhang): the penalty term is
#     φ(x) = (1/2ρ) (‖Π_{S₊ⁿ}(Y - ρ P(x))‖²_F - ‖Y‖²_F)
# with multiplier update Y ← Π_{S₊ⁿ}(Y - ρ P(x)). Gradients flow through the
# analytic rrule of `project_psd`.
module NonconvexAugLagLabSDPExt

using NonconvexAugLagLab
using NonconvexCore: VecModel
import NonconvexAugLagLab: _sd_violation, _sd_terms, _sd_multiplier_update!, _sd_init
using NonconvexSemidefinite: project_psd, psd_infeasibility
using LinearAlgebra

function NonconvexAugLagLab._sd_violation(model::VecModel, i::Int, x::AbstractVector)
    return [psd_infeasibility(model.sd_constraints.fs[i](x))]
end

function NonconvexAugLagLab._sd_init(model::VecModel, sd_rel, ρ)
    return [zeros(sd.dim, sd.dim) for sd in model.sd_constraints.fs[sd_rel]]
end

function NonconvexAugLagLab._sd_terms(model::VecModel, sd_rel, ρ, x, Y)
    fns = model.sd_constraints.fs
    return x -> begin
        val = 0.0
        for (k, i) in enumerate(sd_rel)
            P = fns[i](x)
            Z = project_psd(Y[k] - ρ * P)
            val += (sum(abs2, Z) - sum(abs2, Y[k])) / (2ρ)
        end
        return val
    end
end

function NonconvexAugLagLab._sd_multiplier_update!(model::VecModel, sd_rel, ρ, x, Y)
    for (k, i) in enumerate(sd_rel)
        P = model.sd_constraints.fs[i](x)
        Y[k] = project_psd(Y[k] - ρ * P)
    end
    return Y
end

end
