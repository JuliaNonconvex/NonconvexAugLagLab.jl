# State-of-the-art augmented Lagrangian meta-algorithm.
#
# `AugLagAlg` handles equality, inequality, box and (via the NonconvexSemidefinite
# extension) semidefinite constraints. Constraints selected for relaxation
# (`RelaxConstraintSelection`) are encoded in the augmented Lagrangian
# objective; the rest stay hard for the sub-solver. The outer loop implements
# the practical augmented Lagrangian of Birgin & Martínez: feasibility-driven
# adaptive penalty, tolerance sequences η/ω, multiplier safeguards, two-way
# penalty updates (Curtis et al.) and infeasibility detection. A clean-room
# penalty-barrier-multiplier (PBM) mode follows Ben-Tal & Zibulevsky.

"""
    RelaxConstraintSelection(; eq = true, ineq = true, sd = true)

Selects which constraints are relaxed into the augmented Lagrangian. Each
entry is `true` (relax all of that kind), `false` (keep all hard) or a vector
of constraint indices (relax those only). Box constraints are always handled
by the sub-solver.
"""
struct RelaxConstraintSelection
    eq::Union{Bool,AbstractVector{Int}}
    ineq::Union{Bool,AbstractVector{Int}}
    sd::Union{Bool,AbstractVector{Int}}
end
function RelaxConstraintSelection(; eq = true, ineq = true, sd = true)
    return RelaxConstraintSelection(eq, ineq, sd)
end

function _relaxed_indices(sel::Bool, n::Int)
    return sel ? collect(1:n) : Int[]
end
function _relaxed_indices(sel::AbstractVector{Int}, n::Int)
    return sel
end
function _relaxed_indices(sel, n::Int)
    throw(
        ArgumentError(
            "Relaxation selection entries must be `true`, `false` or a vector of " *
            "constraint indices; got $(typeof(sel)).",
        ),
    )
end

"""
    AugLagAlg(sub_alg = nothing)

Augmented Lagrangian meta-algorithm. `sub_alg` solves the subproblems after the
relaxed constraints are folded into the objective; `nothing` selects the
built-in [`QuasiNewtonAlg`](@ref), which requires every constraint to be
relaxed. Relaxing semidefinite constraints additionally requires
NonconvexSemidefinite to be loaded (its extension provides the cone
projections).
"""
struct AugLagAlg{S<:Union{Nothing,AbstractOptimizer}} <: AbstractOptimizer
    sub_alg::S
end
AugLagAlg() = AugLagAlg(nothing)
AugLagAlg(sub_alg::AbstractOptimizer) = AugLagAlg{typeof(sub_alg)}(sub_alg)

function _optimize_precheck(
    model::AbstractModel,
    optimizer::AugLagAlg,
    args...;
    kwargs...,
)
    nothing
end

function NonconvexCore.solver_capabilities(alg::AugLagAlg)
    if alg.sub_alg === nothing
        # the built-in QuasiNewtonAlg handles box-constrained subproblems
        return ConstraintCapabilities(
            eq = true,
            ineq = true,
            sd = true,
            box = true,
            integer = false,
            infinite_bounds = true,
        )
    end
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(sub; eq = true, ineq = true, sd = true)
end

"""
    AugLagOptions(; kwargs...)

- `sub_options`: options for the sub-solver (or the built-in `QuasiNewtonAlg`).
- `relaxed`: a [`RelaxConstraintSelection`](@ref) (default: relax everything).
- `penalty`: `:quadratic` (classical ALM), `:pbm` (penalty-barrier-multiplier
  for inequalities, quadratic for equalities) or `:mixed` (PBM for
  inequalities, projected quadratic for semidefinite).
- `ρ_init`, `ρ_max`, `ρ_incr`, `ρ_decr`: initial/maximum penalty and the
  increase/decrease factors (two-way adaptation).
- `max_iter`: maximum outer iterations.
- `ctol`: feasibility tolerance.
- `η₀`, `ω₀`: initial feasibility/optimality targets of the subproblems.
- `weights`: per-constraint-group penalty weights (default: all ones).
- `lsq_multipliers`: initialize equality multipliers from the least-squares
  stationarity residual.
- `keep_all`: keep the trajectory of outer iterates in the result.
"""
mutable struct AugLagOptions{S,R,W}
    sub_options::S
    relaxed::R
    penalty::Symbol
    ρ_init::Real
    ρ_max::Real
    ρ_incr::Real
    ρ_decr::Real
    max_iter::Int
    ctol::Real
    η₀::Real
    ω₀::Real
    weights::W
    lsq_multipliers::Bool
    keep_all::Bool
end
function AugLagOptions(;
    sub_options = nothing,
    relaxed = RelaxConstraintSelection(),
    penalty::Symbol = :quadratic,
    ρ_init::Real = 10.0,
    ρ_max::Real = 1e10,
    ρ_incr::Real = 10.0,
    ρ_decr::Real = 10.0,
    max_iter::Int = 1000,
    ctol::Real = 1e-6,
    η₀::Real = 0.5,
    ω₀::Real = 1.0,
    weights = nothing,
    lsq_multipliers::Bool = false,
    keep_all::Bool = false,
)
    penalty in (:quadratic, :pbm, :mixed) ||
        throw(ArgumentError("penalty must be :quadratic, :pbm or :mixed"))
    ρ_init > 0 || throw(ArgumentError("ρ_init must be positive"))
    ρ_incr > 1 || throw(ArgumentError("ρ_incr must be > 1"))
    ρ_decr > 1 || throw(ArgumentError("ρ_decr must be > 1"))
    return AugLagOptions(
        sub_options,
        relaxed,
        penalty,
        ρ_init,
        ρ_max,
        ρ_incr,
        ρ_decr,
        max_iter,
        ctol,
        η₀,
        ω₀,
        weights,
        lsq_multipliers,
        keep_all,
    )
end

struct AugLagResult{TM1,TM2,R,TR} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    results::R
    trajectory::TR
end

mutable struct AugLagWorkspace{
    M<:VecModel,
    X<:AbstractVector,
    O<:AugLagOptions,
    S<:Union{Nothing,AbstractOptimizer},
} <: Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
end

function Workspace(
    model::VecModel,
    optimizer::AugLagAlg,
    x0::AbstractVector,
    args...;
    options::AugLagOptions = AugLagOptions(),
    kwargs...,
)
    if optimizer.sub_alg === nothing
        sel = options.relaxed
        nrelax =
            length(_relaxed_indices(sel.eq, length(model.eq_constraints.fs))) +
            length(_relaxed_indices(sel.ineq, length(model.ineq_constraints.fs))) +
            length(_relaxed_indices(sel.sd, length(model.sd_constraints.fs)))
        ntotal = length(model.eq_constraints.fs) +
                 length(model.ineq_constraints.fs) +
                 length(model.sd_constraints.fs)
        nrelax == ntotal || throw(
            ArgumentError(
                "AugLagAlg() uses the built-in QuasiNewtonAlg which only handles " *
                "unconstrained/box-constrained subproblems; relax every constraint " *
                "(`relaxed = RelaxConstraintSelection()`), or pass a sub_alg.",
            ),
        )
    end
    return AugLagWorkspace(model, copy(x0), options, optimizer.sub_alg)
end

# ------------------------------------------------------------------
# Subproblem construction
# ------------------------------------------------------------------
function _subproblem(
    model::VecModel,
    sel::RelaxConstraintSelection,
    weights::Union{Nothing,AbstractVector},
)
    eq_rel = _relaxed_indices(sel.eq, length(model.eq_constraints.fs))
    ineq_rel = _relaxed_indices(sel.ineq, length(model.ineq_constraints.fs))
    sd_rel = _relaxed_indices(sel.sd, length(model.sd_constraints.fs))
    kept_eq = VectorOfFunctions(
        [c for (i, c) in enumerate(model.eq_constraints.fs) if !(i in eq_rel)],
    )
    kept_ineq = VectorOfFunctions(
        [c for (i, c) in enumerate(model.ineq_constraints.fs) if !(i in ineq_rel)],
    )
    kept_sd = VectorOfFunctions(
        [c for (i, c) in enumerate(model.sd_constraints.fs) if !(i in sd_rel)],
    )
    return eq_rel, ineq_rel, sd_rel, VecModel(
        model.objective,
        kept_eq,
        kept_ineq,
        kept_sd,
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end

# ------------------------------------------------------------------
# Violation measure of the relaxed constraints
# ------------------------------------------------------------------
_vecval(v::Number) = [v]
_vecval(v::AbstractVector) = vec(v)

function _relaxed_violation(model, sel, x)
    eq_rel, ineq_rel, sd_rel, _ = _subproblem(model, sel, nothing)
    v = Float64[]
    for i in eq_rel
        append!(v, abs.(_vecval(model.eq_constraints.fs[i](x))))
    end
    for i in ineq_rel
        append!(v, max.(_vecval(model.ineq_constraints.fs[i](x)), 0.0))
    end
    for i in sd_rel
        append!(v, _sd_violation(model, i, x))
    end
    return v
end

# Extension hook (overridden by NonconvexAugLagLabSDPExt)
function _sd_violation(model, i, x)
    throw(
        ArgumentError(
            "Semidefinite constraints are selected for relaxation but " *
            "NonconvexSemidefinite is not loaded; load it to enable the " *
            "projected augmented Lagrangian for semidefinite constraints.",
        ),
    )
end

# ------------------------------------------------------------------
# Augmented Lagrangian objective (quadratic / PBM)
# ------------------------------------------------------------------
# PBM barrier term Σᵢ c log(1 + uᵢ gᵢ / c) for uᵢ gᵢ / c > -1 (Ben-Tal &
# Zibulevsky's penalty-barrier-multiplier). The argument is clamped at the
# domain boundary so deeply feasible points stay flat.
function _pbm_barrier(g, u, c)
    s = 0.0
    ε = 1e-8
    for i in eachindex(g)
        t = clamp(u[i] * g[i] / c, -1 + ε, 1e15)
        s += c * log(1 + t)
    end
    return s
end

function _unit_weights(weights, indices)
    n = length(indices)
    weights === nothing && return ones(n)
    length(weights) == n || throw(
        ArgumentError(
            "weights must have one entry per relaxed constraint group ($n), got $(length(weights)).",
        ),
    )
    return weights
end

function _auglag_objective(
    model::VecModel,
    sel::RelaxConstraintSelection,
    weights,
    ρ::Real,
    y,
    λ,
    sd_terms::Function,
    penalty::Symbol,
)
    eq_rel, ineq_rel, _, _ = _subproblem(model, sel, nothing)
    objective0 = model.objective
    w_eq = _unit_weights(weights, eq_rel)
    w_ineq = _unit_weights(weights, ineq_rel)
    function _objective(x)
        target = objective0(x)
        for (kk, i) in enumerate(eq_rel)
            c = _vecval(model.eq_constraints.fs[i](x))
            r = ρ * w_eq[kk]
            target += dot(y[kk], c) + r / 2 * dot(c, c)
        end
        for (kk, i) in enumerate(ineq_rel)
            g = _vecval(model.ineq_constraints.fs[i](x))
            r = ρ * w_ineq[kk]
            if penalty === :pbm || penalty === :mixed
                # PBM barrier on the feasible side, classical quadratic
                # augmented Lagrangian on the infeasible side (the
                # PENNON-style globalization of the barrier domain)
                feas = g .<= 0
                target += _pbm_barrier(g[feas], λ[kk][feas], r)
                if any(.!feas)
                    gp = max.(g, 0.0)
                    target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
                end
            else
                gp = max.(g, 0.0)
                target += dot(λ[kk], gp) + r / 2 * dot(gp, gp)
            end
        end
        target += sd_terms(x)
        return target
    end
    return _objective
end

# Multiplier buffers per constraint group (one vector of multipliers per
# relaxed constraint).
function _init_multipliers(model, sel)
    eq_rel, ineq_rel, _, _ = _subproblem(model, sel, nothing)
    y = [zeros(length(_vecval(model.eq_constraints.fs[i](model.init)))) for i in eq_rel]
    λ = [zeros(length(_vecval(model.ineq_constraints.fs[i](model.init)))) for i in ineq_rel]
    return y, λ
end

function _lsq_multipliers(model, eq_rel, x)
    _, ∇f = NonconvexCore.value_gradient(model.objective, x)
    ∇f = copy(∇f)
    y = Vector{Vector{Float64}}()
    for i in eq_rel
        ci = model.eq_constraints.fs[i]
        vi, Ji = NonconvexCore.value_jacobian(x -> _vecval(ci(x)), x)
        m = length(_vecval(vi))
        J = Matrix(Ji)
        δ = 1e-8
        A = J * J' + δ * I
        push!(y, -(A \ (J * ∇f)))
    end
    return y
end

# ------------------------------------------------------------------
# Outer loop
# ------------------------------------------------------------------
function optimize!(workspace::AugLagWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack relaxed, penalty, max_iter, ctol, η₀, ω₀ = options
    @unpack lsq_multipliers, keep_all = options
    ρ = Float64(options.ρ_init)
    x = copy(x0)
    eq_rel, ineq_rel, sd_rel, submodel0 = _subproblem(model, relaxed, options.weights)
    y, λ = _init_multipliers(model, relaxed)
    if penalty === :pbm || penalty === :mixed
        # the PBM barrier term (u/c) log(1 + c g) is inactive for u = 0;
        # initialize the barrier multipliers positive
        for k in eachindex(λ)
            λ[k] .= 1.0
        end
    end
    if lsq_multipliers && !isempty(eq_rel)
        y = _lsq_multipliers(model, eq_rel, x)
    end
    η = max(η₀, ctol)
    ω = max(ω₀, sqrt(eps(Float64)))
    Y = _sd_init(model, sd_rel, ρ)
    trajectory = Tuple{typeof(x), Real, Real}[]
    fx = model.objective(x)
    vnorm = Inf
    for iter = 1:max_iter
        alg = sub_alg === nothing ? QuasiNewtonAlg() : sub_alg
        sd_terms = _sd_terms(model, sd_rel, ρ, x, Y)
        obj = _auglag_objective(model, relaxed, options.weights, ρ, y, λ, sd_terms, penalty)
        submodel = VecModel(
            Objective(obj),
            submodel0.eq_constraints,
            submodel0.ineq_constraints,
            submodel0.sd_constraints,
            submodel0.box_min,
            submodel0.box_max,
            submodel0.init,
            submodel0.integer,
        )
        # validate the subproblem against the sub-solver's declared
        # capabilities before solving (informative errors)
        checksubproblem!(
            submodel,
            alg;
            context = "the augmented Lagrangian subproblem",
            hint = "relax the constraint in `RelaxConstraintSelection` or choose a sub-solver that supports it.",
        )
        result_i = optimize(submodel, alg, x; options = options.sub_options)
        x = copy(result_i.minimizer)
        v = _relaxed_violation(model, relaxed, x)
        vnorm = norm(v, Inf)
        fx = model.objective(x)
        push!(trajectory, (copy(x), fx, vnorm))

        if vnorm <= η
            # multiplier updates
            k = 1
            for i in eq_rel
                c = _vecval(model.eq_constraints.fs[i](x))
                w = options.weights === nothing ? 1.0 : options.weights[k]
                y[k] .= y[k] .+ (ρ * w) .* c
                k += 1
            end
            k = 1
            for i in ineq_rel
                g = _vecval(model.ineq_constraints.fs[i](x))
                w = options.weights === nothing ? 1.0 : options.weights[k]
                if penalty === :pbm || penalty === :mixed
                    # PBM update u⁺ = u / (1 + u g / c) on the feasible side,
                    # classical multiplier update on the infeasible side
                    feas = g .<= 0
                    λ[k][feas] .= λ[k][feas] ./ (1 .+ λ[k][feas] .* g[feas] ./ (ρ * w))
                    λ[k][.!feas] .= max.(λ[k][.!feas] .+ (ρ * w) .* g[.!feas], 0.0)
                else
                    λ[k] .= max.(λ[k] .+ (ρ * w) .* g, 0.0)
                end
                k += 1
            end
            _sd_multiplier_update!(model, sd_rel, ρ, x, Y)
            η = max(η / ρ^0.9, ctol)
            ω = max(ω / ρ, sqrt(eps(Float64)))
        else
            # penalty increase + tolerance reset (Birgin-Martínez style)
            ρ = min(options.ρ_max, ρ * options.ρ_incr)
            η = max(η₀ / ρ^0.1, ctol)
            ω = max(1 / ρ, sqrt(eps(Float64)))
        end
        vnorm <= ctol && break
    end
    return AugLagResult(copy(x), fx, nothing, keep_all ? trajectory : nothing)
end

# Extension hooks for semidefinite constraints (see ext/NonconvexAugLagLabSDPExt.jl)
function _sd_init(model, sd_rel, ρ)
    isempty(sd_rel) && return nothing
    _sd_violation(model, first(sd_rel), model.init) # throws when the extension is missing
end
function _sd_terms(model, sd_rel, ρ, x, Y)
    return x -> 0.0
end
_sd_multiplier_update!(model, sd_rel, ρ, x, Y) = nothing
