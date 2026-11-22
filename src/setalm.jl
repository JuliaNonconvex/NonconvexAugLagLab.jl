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
using NonconvexCore:
    Reals, EqualTo, Zeros, LessThan, GreaterThan, Interval
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

# ------------------------------------------------------------------
# Violation (bundle evaluated once; value-only, so mutation is fine)
# ------------------------------------------------------------------

function _set_violation(model::VecModel, i::Int, x)
    c = model.set_constraints.fs[i]
    v, comps = _set_components(c, x)
    out = Float64[]
    for (vi, s) in zip(v, comps)
        s isa Reals && continue
        if _set_is_psd(s)
            append!(out, _set_psd_violation(vi))
        else
            append!(out, _component_violation(vi, s))
        end
    end
    return out
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

function _set_terms(model::VecModel, i::Int, ρ, x, mults, penalty::Symbol)
    c = model.set_constraints.fs[i]
    y, λ, Y = mults
    v, comps = _set_components(c, x)
    s = 0.0
    k_eq = 1
    k_ineq = 1
    k_psd = 1
    for (vi, set) in zip(v, comps)
        if set isa Reals
            continue
        elseif _set_is_psd(set)
            s += _set_psd_term(Y[k_psd], ρ, vi, penalty)
            k_psd += 1
        elseif set isa _SET_EQ
            g = _component_violation(vi, set)
            m = _component_dim(set)
            yy = @view y[k_eq:(k_eq + m - 1)]
            s += dot(yy, g) + (ρ / 2) * dot(g, g)
            k_eq += m
        else
            g = _component_violation(vi, set)
            m = _component_dim(set)
            λλ = @view λ[k_ineq:(k_ineq + m - 1)]
            if penalty in (:pbm, :mixed, :mixed_logdet)
                feas = g .<= 0
                s += _pbm_barrier(g[feas], λλ[feas], ρ)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    s += dot(λλ, gp) + (ρ / 2) * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                s += dot(λλ, gp) + (ρ / 2) * dot(gp, gp)
            end
            k_ineq += m
        end
    end
    return s
end

# The combined penalty term of all relaxed set constraints (a closure, like
# `_sd_terms`).
function _set_terms_closure(model::VecModel, sets_rel, ρ, x, Msets, penalty::Symbol)
    isempty(sets_rel) && return _ -> 0.0
    return z -> begin
        s = 0.0
        for (k, i) in enumerate(sets_rel)
            s += _set_terms(model, i, ρ, z, Msets[k], penalty)
        end
        return s
    end
end

# ------------------------------------------------------------------
# Multiplier updates
# ------------------------------------------------------------------

function _set_update!(model::VecModel, sets_rel, ρ, x, Msets, penalty::Symbol)
    for (k, i) in enumerate(sets_rel)
        c = model.set_constraints.fs[i]
        y, λ, Y = Msets[k]
        v, comps = _set_components(c, x)
        k_eq = 1
        k_ineq = 1
        k_psd = 1
        for (vi, set) in zip(v, comps)
            if set isa Reals
                continue
            elseif _set_is_psd(set)
                _set_psd_update!(Y[k_psd], ρ, vi, penalty)
                k_psd += 1
            elseif set isa _SET_EQ
                g = _component_violation(vi, set)
                m = _component_dim(set)
                y[k_eq:(k_eq + m - 1)] .+= ρ .* g
                k_eq += m
            else
                g = _component_violation(vi, set)
                m = _component_dim(set)
                λ[k_ineq:(k_ineq + m - 1)] .= max.(λ[k_ineq:(k_ineq + m - 1)] .+ ρ .* g, 0.0)
                k_ineq += m
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# PSD-component hooks (overridden by NonconvexAugLagLabSDPExt)
# ------------------------------------------------------------------

function _set_psd_violation(vi)
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

_set_psd_term(Y, ρ, vi, penalty) = throw(
    ArgumentError(
        "Semidefinite components of set constraints are selected for " *
        "relaxation but NonconvexSemidefinite is not loaded; load it to " *
        "enable the projected augmented Lagrangian for semidefinite constraints.",
    ),
)
_set_psd_update!(Y, ρ, vi, penalty) = nothing
