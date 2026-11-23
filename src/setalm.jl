# Function-in-set constraints (NonconvexCore's `FunctionInSet`, with
# SetProducts, PSD components and chance constraints) in the augmented
# Lagrangian.
#
# Design: each relaxed set constraint is handled as ONE group whose bundle
# function is evaluated exactly once per call — the displacement solve and
# assemblies behind a `SetProduct` constraint run once, then the components
# are charged with the classical ALM terms:
#   EqualTo/Zeros          → equality terms       (quadratic ALM),
#   LessThan/GreaterThan/Interval/Nonnegatives/Nonpositives/Simplex/ChanceSet
#                          → inequality terms     (quadratic/PBM),
#   PSDCone                → projected ALM terms  (NonconvexSemidefinite ext),
#   Reals                  → skipped (vacuous).
# Set constraints kept hard (`RelaxConstraintSelection(sets = ...)`) stay in
# the subproblem and are validated against the sub-solver's `sets`
# capabilities.

using NonconvexCore: set_value, components, _component_violation
using NonconvexCore: Reals, EqualTo, Zeros, LessThan, GreaterThan, Interval
using NonconvexCore: Nonnegatives, Nonpositives, Simplex, ChanceSet, SetProduct

# ------------------------------------------------------------------
# Component bookkeeping
# ------------------------------------------------------------------

# Whether a component set is a semidefinite cone: the PSDCone type lives in
# NonconvexSemidefinite.jl, whose extension of AugLagLab specializes this
# hook (and `_component_dim`).
_set_is_psd(s) = false

# The length of the violation vector of a component in a set.
_component_dim(s::Simplex) = s.dim + 1
_component_dim(s::ChanceSet) = length(components(s.set))
_component_dim(s::Union{EqualTo,Zeros}) = s.dim
_component_dim(s::Union{LessThan,GreaterThan,Interval,Nonnegatives,Nonpositives}) = s.dim
_component_dim(s::Reals) = 0
_component_dim(s::SetProduct) = sum(_component_dim, s.sets; init=0)

const _SET_EQ = Union{EqualTo,Zeros}
const _SET_INEQ = Union{
    LessThan,GreaterThan,Interval,Nonnegatives,Nonpositives,Simplex,ChanceSet
}

# The component values of one bundle evaluation, as a tuple.
function _set_components(c::FunctionInSet, x)
    v, comps = set_value(c, x), components(c.set)
    return v, comps
end

# One scale per component of a set constraint: the component's reference
# magnitude max |vᵢ(x₀)|, floored at 1. A value-based normalization makes
# the residuals unit-free while keeping the penalty weights meaningful; a
# gradient probe is unsuitable here because the components (e.g. stiffness
# matrices assembled through a linear solve) can have derivative magnitudes
# unrelated to their values.
function _set_scales(model::VecModel, i::Int, x0, scale_mode::Symbol)
    c = model.set_constraints.fs[i]
    comps = NonconvexCore.components(c.set)
    if scale_mode === :value
        # Value-based (the default). Order components are scaled by their
        # reference magnitude. Equality components (EqualTo/Zeros) instead
        # use the gradient-magnitude probe: their values track functions
        # whose derivatives can exceed their value magnitudes by orders of
        # magnitude (e.g. a coupling residual of a matrix assembled through
        # a linear solve), and a value scale would under-normalize the
        # penalty's x-gradient and let it yank the iterate away.
        δ = 1e-6 .* (1 .+ abs.(x0))
        δnorm = norm(δ)
        v1 = NonconvexCore.set_value(c, x0 .+ δ)
        v0 = NonconvexCore.set_value(c, x0)
        return [
            begin
                if s isa Union{EqualTo,Zeros}
                    d = norm(a .- b) / δnorm
                    (isfinite(d) && d >= 1.0) ? d : 1.0
                else
                    sc = maximum(abs, a)
                    (isfinite(sc) && sc >= 1.0) ? sc : 1.0
                end
            end for (a, b, s) in zip(v0, v1, comps)
        ]
    end
    # Gradient-magnitude probes for every component (opt-in mode).
    δ = 1e-6 .* (1 .+ abs.(x0))
    δnorm = norm(δ)
    v1 = NonconvexCore.set_value(c, x0 .+ δ)
    v0 = NonconvexCore.set_value(c, x0)
    return [max(1.0, norm(a .- b) / δnorm) for (a, b) in zip(v0, v1)]
end

# ------------------------------------------------------------------
# Violation (bundle evaluated once; value-only, so mutation is fine)
# ------------------------------------------------------------------

function _set_violation(model::VecModel, i::Int, x, scales=nothing)
    c = model.set_constraints.fs[i]
    v, comps = _set_components(c, x)
    # Mutation-free: the violation is also the restoration-phase merit,
    # which the sub-solver differentiates.
    sc = scales === nothing ? fill(1.0, length(v)) : scales
    vs = [
        if s isa Reals
            Float64[]
        elseif _set_is_psd(s)
            _set_psd_violation(vi, sc[k])
        else
            _component_violation(vi, s) ./ sc[k]
        end for (k, (vi, s)) in enumerate(zip(v, comps))
    ]
    return vcat(vs...)
end

# ------------------------------------------------------------------
# Multipliers: (y_eq, λ_ineq, Y_psd) per relaxed set constraint
# ------------------------------------------------------------------

function _set_init_multipliers(model::VecModel, i::Int, ρ)
    c = model.set_constraints.fs[i]
    comps = components(c.set)
    ne = sum(_component_dim(s) for s in comps if s isa _SET_EQ; init=0)
    ni = sum(_component_dim(s) for s in comps if s isa _SET_INEQ; init=0)
    Y = any(_set_is_psd, comps) ? _set_psd_init(c, ρ) : Matrix{Float64}[]
    return zeros(ne), zeros(ni), Y
end

# ------------------------------------------------------------------
# The augmented-Lagrangian penalty term of one set constraint
# (bundle evaluated once)
# ------------------------------------------------------------------

function _set_terms(model::VecModel, i::Int, ρ, x, mults, penalty::Symbol, scales, w)
    c = model.set_constraints.fs[i]
    y, λ, Y = mults
    v, comps = _set_components(c, x)
    # The PSD penalty runs on the RAW matrices: the projected augmented
    # Lagrangian is scale-adaptive (its projection absorbs the matrix
    # scale), and dividing P by a value scale would suppress the penalty
    # force by sc². Only the eq/ineq components use the value scales, where
    # the quadratic penalty needs them. The group weight w scales the
    # penalty parts (multiplier terms stay unweighted, as in the classical
    # groups).
    s = _set_psd_terms(Y, ρ * w, v, comps, fill(1.0, length(comps)), penalty)
    k_eq = 1
    k_ineq = 1
    for (k, (vi, set)) in enumerate(zip(v, comps))
        (set isa Reals || _set_is_psd(set)) && continue
        sc = scales[k]
        if set isa _SET_EQ
            g = _component_violation(vi, set) ./ sc
            m = _component_dim(set)
            yy = @view y[k_eq:(k_eq + m - 1)]
            s += dot(yy, g) + (ρ * w / 2) * dot(g, g)
            k_eq += m
        else
            g = _component_violation(vi, set) ./ sc
            m = _component_dim(set)
            λλ = @view λ[k_ineq:(k_ineq + m - 1)]
            if penalty in (:pbm, :mixed, :mixed_logdet)
                feas = g .<= 0
                s += _pbm_barrier(g[feas], λλ[feas], ρ * w)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    s += dot(λλ, gp) + (ρ * w / 2) * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                s += dot(λλ, gp) + (ρ * w / 2) * dot(gp, gp)
            end
            k_ineq += m
        end
    end
    return s
end

# The combined penalty term of all relaxed set constraints (a closure, like
# `_sd_terms`).
function _set_terms_closure(
    model::VecModel, sets_rel, ρ, x, Msets, penalty::Symbol, set_scales, w_sets
)
    isempty(sets_rel) && return _ -> 0.0
    return z -> begin
        s = 0.0
        for (k, i) in enumerate(sets_rel)
            s += _set_terms(model, i, ρ, z, Msets[k], penalty, set_scales[k], w_sets[k])
        end
        return s
    end
end

# ------------------------------------------------------------------
# Multiplier updates
# ------------------------------------------------------------------

function _set_update!(
    model::VecModel,
    sets_rel,
    ρ,
    x,
    Msets,
    penalty::Symbol,
    set_scales,
    w_sets,
    θ_mult,
    safeguard::Bool,
    set_and_hist,
    anderson_ineq::Bool,
    anderson_depth::Int,
)
    use_and_eq = penalty === :quadratic
    use_and_ineq = anderson_ineq && penalty === :quadratic
    for (k, i) in enumerate(sets_rel)
        c = model.set_constraints.fs[i]
        y, λ, Y = Msets[k]
        v, comps = _set_components(c, x)
        sc = set_scales[k]
        w = w_sets[k]
        hist_eq, hist_ineq = set_and_hist[k]
        k_eq = 1
        k_ineq = 1
        k_psd = 1
        for (k2, (vi, set)) in enumerate(zip(v, comps))
            if set isa Reals
                continue
            elseif _set_is_psd(set)
                # raw matrices: the projected ALM is scale-adaptive
                _set_psd_update!(Y[k_psd], ρ * w * θ_mult, vi, penalty)
                k_psd += 1
            elseif set isa _SET_EQ
                g = _component_violation(vi, set) ./ sc[k2]
                m = _component_dim(set)
                yk = y[k_eq:(k_eq + m - 1)]
                step = θ_mult .* (ρ * w) .* g
                if use_and_eq && !isempty(hist_eq)
                    # Anderson acceleration of the set-equality multiplier map
                    yk .= _anderson_mix(hist_eq, yk .+ step)
                    push!(hist_eq, (copy(yk), step))
                    length(hist_eq) > anderson_depth && popfirst!(hist_eq)
                else
                    yk .+= step
                    push!(hist_eq, (copy(yk), step))
                    length(hist_eq) > anderson_depth && popfirst!(hist_eq)
                end
                k_eq += m
            else
                g = _component_violation(vi, set) ./ sc[k2]
                m = _component_dim(set)
                λk = λ[k_ineq:(k_ineq + m - 1)]
                step = θ_mult .* (ρ * w) .* g
                if use_and_ineq && !isempty(hist_ineq)
                    λk .= _anderson_mix(hist_ineq, λk .+ step)
                    push!(hist_ineq, (copy(λk), step))
                    length(hist_ineq) > anderson_depth && popfirst!(hist_ineq)
                else
                    λk .+= step
                    push!(hist_ineq, (copy(λk), step))
                    length(hist_ineq) > anderson_depth && popfirst!(hist_ineq)
                end
                λk .= max.(λk, 0.0)
                if safeguard
                    λk .= min.(λk, 1e8)
                end
                k_ineq += m
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# PSD-component hooks (overridden by NonconvexAugLagLabSDPExt)
# ------------------------------------------------------------------

function _set_psd_violation(vi, sc)
    return throw(
        ArgumentError(
            "Semidefinite components of set constraints are selected for " *
            "relaxation but NonconvexSemidefinite is not loaded; load it to " *
            "enable the projected augmented Lagrangian for semidefinite " *
            "constraints.",
        ),
    )
end

function _set_psd_init(c, ρ)
    return throw(
        ArgumentError(
            "Semidefinite components of set constraints are selected for " *
            "relaxation but NonconvexSemidefinite is not loaded; load it to " *
            "enable the projected augmented Lagrangian for semidefinite " *
            "constraints.",
        ),
    )
end

function _set_psd_term(Y, ρ, vi, penalty)
    return throw(
        ArgumentError(
            "Semidefinite components of set constraints are selected for " *
            "relaxation but NonconvexSemidefinite is not loaded; load it to " *
            "enable the projected augmented Lagrangian for semidefinite constraints.",
        ),
    )
end

function _set_psd_terms(Y, ρ, v::Tuple, comps::Tuple, penalty)
    return throw(
        ArgumentError(
            "Semidefinite components of set constraints are selected for " *
            "relaxation but NonconvexSemidefinite is not loaded; load it to " *
            "enable the projected augmented Lagrangian for semidefinite constraints.",
        ),
    )
end
_set_psd_update!(Y, ρ, vi, penalty) = nothing

function _set_psd_multiplier_terms(Y, v::Tuple, comps::Tuple)
    return throw(
        ArgumentError(
            "Semidefinite components of set constraints are selected for " *
            "relaxation but NonconvexSemidefinite is not loaded; load it to " *
            "enable the projected augmented Lagrangian for semidefinite constraints.",
        ),
    )
end

# ------------------------------------------------------------------
# Multiplier terms and complementarity of the set constraints: the
# stationarity contributions ⟨y, g_eq⟩ + ⟨λ, g_ineq⁺⟩ + Σₖ⟨Yₖ, Pₖ⟩ and
# max |λⱼ gⱼ⁺| used by the KKT-complete stopping test. The PSD part is a
# single batched call (the multiplier terms are linear in both arguments).
# ------------------------------------------------------------------

function _set_multiplier_terms(model::VecModel, i::Int, x, mults, scales)
    c = model.set_constraints.fs[i]
    y, λ, Y = mults
    v, comps = _set_components(c, x)
    s = _set_psd_multiplier_terms(Y, v, comps, fill(1.0, length(comps)))
    k_eq = 1
    k_ineq = 1
    for (k, (vi, set)) in enumerate(zip(v, comps))
        if set isa Reals
            continue
        elseif _set_is_psd(set)
            continue
        elseif set isa _SET_EQ
            g = _component_violation(vi, set) ./ scales[k]
            m = _component_dim(set)
            yy = @view y[k_eq:(k_eq + m - 1)]
            s += dot(yy, g)
            k_eq += m
        else
            g = _component_violation(vi, set) ./ scales[k]
            m = _component_dim(set)
            λλ = @view λ[k_ineq:(k_ineq + m - 1)]
            s += dot(λλ, max.(g, 0.0))
            k_ineq += m
        end
    end
    return s
end

function _set_complementarity(model::VecModel, i::Int, x, mults, scales)
    c = model.set_constraints.fs[i]
    _, λ, _ = mults
    v, comps = _set_components(c, x)
    comp = 0.0
    k_ineq = 1
    for (k, (vi, set)) in enumerate(zip(v, comps))
        (set isa Reals || _set_is_psd(set) || set isa _SET_EQ) && continue
        g = _component_violation(vi, set) ./ scales[k]
        m = _component_dim(set)
        λλ = @view λ[k_ineq:(k_ineq + m - 1)]
        comp = max(comp, maximum(abs.(λλ .* max.(g, 0.0))))
        k_ineq += m
    end
    return comp
end
