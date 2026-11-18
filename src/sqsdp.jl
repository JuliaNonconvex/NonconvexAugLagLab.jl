# Sequential quadratic SDP (SQSDP, Freund-Jarre-Vogelbusch style, first cut).
#
# At every iterate the constraints are linearized - including the
# semidefinite one, which becomes a linear matrix inequality (LMI) - and a
# quadratic program with an LMI constraint is solved for the step:
#
#     min_d  ∇f(x)'d + ½ d'Bd
#     s.t.   c(x) + Jc d = 0,  g(x) + Jg d ≤ 0,
#            P(x) + DP(x)[d] ⪰ 0,  ‖d‖ ≤ Δ
#
# The subproblem is solved by the barrier meta-algorithm (BarrierAlg +
# Ipopt); like BarrierAlg it requires the sd iterate strictly feasible
# (the LMI at d = 0 is P(x) ≻ 0). The step is globalized by the augmented
# Lagrangian merit function with L-BFGS curvature (quasi-SQSDP; exact
# second-order curvature is future work). Semidefinite support needs
# NonconvexSemidefinite loaded.

# --- options and result ------------------------------------------------------

mutable struct SQSDPOptions
    max_iter::Int
    ρ::Real                # AL merit penalty
    Δ0::Real               # initial trust region
    Δmax::Real
    η::Real                # ared/pred acceptance ratio
    ctol::Real
    memory::Int            # L-BFGS memory for the curvature
    sub_solver::Any        # solver of the QP-with-LMI subproblem (required)
    sub_options::Any       # its options
    keep_all::Bool
end
function SQSDPOptions(;
    max_iter::Int=50,
    ρ::Real=10.0,
    Δ0::Real=1.0,
    Δmax::Real=10.0,
    η::Real=0.1,
    ctol::Real=1e-6,
    memory::Int=10,
    sub_solver=nothing,
    sub_options=nothing,
    keep_all::Bool=false,
)
    return SQSDPOptions(
        max_iter, ρ, Δ0, Δmax, η, ctol, memory, sub_solver, sub_options, keep_all
    )
end

struct SQSDPResult{TM1,TM2,R,TR} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    results::R
    trajectory::TR
end

struct SQSDPAlg <: AbstractOptimizer end

function NonconvexCore.solver_capabilities(::SQSDPAlg)
    return ConstraintCapabilities(;
        linear_eq=true,
        nonlinear_eq=true,
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=true,
        nonlinear_sd=true,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=true,
    )
end

mutable struct SQSDPWorkspace{M<:VecModel,X<:AbstractVector,O<:SQSDPOptions} <: Workspace
    model::M
    x0::X
    options::O
    state::Any
end

function Workspace(
    model::VecModel,
    optimizer::SQSDPAlg,
    x0::AbstractVector,
    args...;
    options::SQSDPOptions=SQSDPOptions(),
    kwargs...,
)
    return SQSDPWorkspace(model, copy(x0), options, nothing)
end

# --- merit and helpers --------------------------------------------------------

function _sqsdp_merit(model, x, ρ)
    f = model.objective(x)
    v = sum(c -> sum(abs2, _vecval(c(x))), model.eq_constraints.fs; init=0.0)
    v += sum(c -> sum(abs2, max.(_vecval(c(x)), 0.0)), model.ineq_constraints.fs; init=0.0)
    for (i, c) in enumerate(model.sd_constraints.fs)
        v += sum(abs2, _sd_violation(model, i, x))
    end
    return f + (ρ / 2) * v
end

function _sqsdp_violation(model, x)
    v = 0.0
    for c in model.eq_constraints.fs
        v = max(v, norm(_vecval(c(x)), Inf))
    end
    for c in model.ineq_constraints.fs
        v = max(v, maximum(max.(_vecval(c(x)), 0.0)))
    end
    for i in eachindex(model.sd_constraints.fs)
        v = max(v, maximum(_sd_violation(model, i, x)))
    end
    return v
end

function _sqsdp_hv!(Hv, approx, d)
    fill!(Hv, 0.0)
    apply!(Hv, approx, d)
    return Hv
end

# sd multiplier projection via the extension; without NonconvexSemidefinite
# loaded the fallback is `nothing` (the sd violation check throws first, but
# keep this path informative for JET and direct calls).
function _sd_proj_update(model, k, ρ, x, Yk)
    Ys = _sd_multiplier_update!(model, [k], ρ, x, [Yk], :quadratic)
    Ys === nothing && throw(
        ArgumentError(
            "SQSDP with semidefinite constraints requires " *
            "NonconvexSemidefinite to be loaded",
        ),
    )
    return Ys[1]
end

# --- driver ------------------------------------------------------------------

function optimize!(workspace::SQSDPWorkspace)
    @unpack model, options = workspace
    x = copy(workspace.x0)
    @unpack max_iter, ρ, Δ0, Δmax, η, ctol, memory, keep_all = options
    sub_solver = options.sub_solver
    sub_solver === nothing && throw(
        ArgumentError(
            "SQSDP requires a sub-solver for the QP-with-LMI subproblem, e.g. " *
            "SQSDPOptions(sub_solver = BarrierAlg(IpoptAlg()), sub_options = " *
            "BarrierOptions(sub_options = IpoptOptions())) (load " *
            "NonconvexSemidefinite and NonconvexIpopt).",
        ),
    )
    obj = model.objective
    obj === nothing && throw(ArgumentError("The model has no objective."))
    n = length(x)
    eq_fs = model.eq_constraints.fs
    ineq_fs = model.ineq_constraints.fs
    sd_fs = model.sd_constraints.fs
    y = [zeros(length(_vecval(c(x)))) for c in eq_fs]
    λ = [zeros(length(_vecval(c(x)))) for c in ineq_fs]
    Y = [zeros(sd.dim, sd.dim) for sd in sd_fs]
    approx = LBFGSApprox(memory; H0=IdentityPreconditioner())
    Δ = Float64(Δ0)
    trajectory = Tuple{typeof(x),Real,Real}[]
    f_new = obj(x)
    for iter in 1:max_iter
        f_new, g = NonconvexCore.value_gradient(obj, x)
        g = copy(g)
        # the L-BFGS curvature is fixed within the iteration: materialize it
        # densely so the QP objective is a plain quadratic (AD-friendly)
        Bmat = zeros(n, n)
        for j in 1:n
            ej = zeros(n)
            ej[j] = 1.0
            Bmat[:, j] .= _sqsdp_hv!(zeros(n), approx, ej)
        end
        # QP-with-LMI subproblem: linearized constraints, L-BFGS curvature
        sub = Model(d -> dot(g, d) + 0.5 * dot(d, Bmat * d))
        addvar!(sub, fill(-Δ, n), fill(Δ, n); init=zeros(n))
        for c in eq_fs
            c0 = _vecval(c(x))
            add_eq_constraint!(sub, d -> begin
                _, J = NonconvexCore.value_jacobian(c, x)
                return c0 + J * d
            end)
        end
        for c in ineq_fs
            c0 = _vecval(c(x))
            add_ineq_constraint!(sub, d -> begin
                _, J = NonconvexCore.value_jacobian(c, x)
                return c0 + J * d
            end)
        end
        for c in sd_fs
            P0 = c(x)
            # Directional derivative of the sd function via central finite
            # differences: Zygote's withjacobian path fails on matrix-valued
            # functions, and the LMI linearization only needs first-order
            # accuracy (documented first-cut limitation).
            h = 1e-7 * (1 + norm(x, Inf))
            add_sd_constraint!(
                sub, d -> begin
                    DPd = (c(x .+ h .* d) .- c(x .- h .* d)) ./ (2h)
                    return DPd .+ P0
                end
            )
        end
        rsub = if options.sub_options === nothing
            optimize(sub, sub_solver, zeros(n))
        else
            optimize(sub, sub_solver, zeros(n); options=options.sub_options)
        end
        d = rsub.minimizer
        # trust-region ratio on the AL merit
        m0 = _sqsdp_merit(model, x, ρ)
        mnew = _sqsdp_merit(model, x .+ d, ρ)
        ared = m0 - mnew
        pred = -(dot(g, d) + 0.5 * dot(d, Bmat * d))
        if pred > 0 && ared / pred > η
            xnew = x .+ d
            Δ = min(Δmax, 2Δ)
            # AL-style multiplier updates
            for k in eachindex(eq_fs)
                y[k] .= y[k] .+ ρ .* _vecval(eq_fs[k](xnew))
            end
            for k in eachindex(ineq_fs)
                λ[k] .= max.(λ[k] .+ ρ .* _vecval(ineq_fs[k](xnew)), 0.0)
            end
            for k in eachindex(sd_fs)
                Y[k] = _sd_proj_update(model, k, ρ, xnew, Y[k])
            end
            # secant pair on the AL merit gradient (assembled by parts:
            # Zygote's sum-over-empty-tuples path breaks on constraint-free
            # groups, and the sd part uses the violation rrule)
            gmerit = copy(g)
            for c in eq_fs
                cv = _vecval(c(xnew))
                _, J = NonconvexCore.value_jacobian(c, xnew)
                gmerit .+= ρ .* (transpose(J) * cv)
            end
            for c in ineq_fs
                cv = max.(_vecval(c(xnew)), 0.0)
                _, J = NonconvexCore.value_jacobian(c, xnew)
                gmerit .+= ρ .* (transpose(J) * cv)
            end
            for (i, c) in enumerate(sd_fs)
                _, gv = NonconvexCore.value_gradient(
                    z -> sum(abs2, _sd_violation(model, i, z)), xnew
                )
                gmerit .+= ρ .* gv
            end
            gnew = gmerit
            s = xnew .- x
            gg = gnew .- g
            if dot(s, gg) > 0
                update!(approx, s, gg)
            end
            x = xnew
            f_new = obj(x)
        else
            Δ = max(Δ / 2, 1e-8)
        end
        v = _sqsdp_violation(model, x)
        push!(trajectory, (copy(x), f_new, v))
        v <= ctol && break
    end
    workspace.state = (; x, y, λ, Y)
    return SQSDPResult(copy(x), obj(x), nothing, keep_all ? trajectory : nothing)
end
