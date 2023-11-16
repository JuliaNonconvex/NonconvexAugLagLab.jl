struct AugLag2{P,D} <: AbstractOptimizer
    primaloptimizer::P
    dualoptimizer::D
end
function AugLag2(;
    primaloptimizer = Optim.ConjugateGradient(
        linesearch = Optim.LineSearches.BackTracking(iterations = 10),
    ),
    dualoptimizer = Optim.GradientDescent(
        linesearch = Optim.LineSearches.BackTracking(iterations = 10),
    ),
)
    return AugLag2(primaloptimizer, dualoptimizer)
end

struct AugLag2Options{P,D,M,T,Q}
    primaloptions::P
    dualoptions::D
    maxiter::M
    tol::T
    quadfactor::Q
end
function AugLag2Options(
    alg::AugLag2;
    primaloptions = alg.primaloptimizer isa MMA02 || alg.primaloptimizer isa MMA87 ?
                    MMAOptions(maxiter = 100, tol = Tolerance(kkt = 1e-4)) :
                    Optim.Options(outer_iterations = 10, iterations = 10),
    dualoptions = Optim.Options(outer_iterations = 10, iterations = 10),
    maxiter = 10,
    tol = Tolerance(),
    quadfactor = 10,
)
    return AugLag2Options(primaloptions, dualoptions, maxiter, tol, quadfactor)
end

function Solution(lagmodel::AugLag2Model)
    prevx = copy(getmin(lagmodel))
    x = copy(prevx)
    prevf = Inf
    f = Inf
    λ = copy(getlinweights(lagmodel))
    g = copy(λ)
    convstate = ConvergenceState()
    return Solution(
        prevx,
        x,
        getlinweights(lagmodel),
        prevf,
        f,
        nothing,
        g,
        nothing,
        convstate,
    )
end

mutable struct AugLag2Workspace{
    M<:VecModel,
    L<:AugLag2Model,
    X<:AbstractVector,
    O1<:AugLag2,
    O2<:AugLag2Options,
    S<:Solution,
    C1<:ConvergenceCriteria,
    C2<:Function,
    T<:Trace,
} <: Workspace
    model::M
    lagmodel::L
    x0::X
    optimizer::O1
    options::O2

    solution::S
    convcriteria::C1
    callback::C2

    trace::T
    outer_iter::Int
    iter::Int
    fcalls::Int
end
function AugLag2Workspace(
    model::VecModel,
    optimizer::AugLag2,
    x0::AbstractVector;
    options::AugLag2Options = AugLag2Options(optimizer),
    convcriteria::ConvergenceCriteria = KKTCriteria(),
    plot_trace::Bool = false,
    show_plot::Bool = plot_trace,
    save_plot = nothing,
    callback::Function = plot_trace ?
                         LazyPlottingCallback(;
        show_plot = show_plot,
        save_plot = save_plot,
    ) : NoCallback(),
    kwargs...,
)
    T = eltype(x0)
    lagmodel = AugLag2Model(model; kwargs...)

    # Convergence
    solution = Solution(lagmodel)
    #assess_convergence!(solution, model, options.tol, convcriteria)

    # Trace
    trace = Trace([])

    # Iteraton counter
    fcalls, outer_iter, iter = 1, 0, 0

    return AugLag2Workspace(
        model,
        lagmodel,
        x0,
        optimizer,
        options,
        solution,
        convcriteria,
        callback,
        trace,
        outer_iter,
        iter,
        fcalls,
    )
end

Workspace(model::VecModel, alg::AugLag2, x0::AbstractVector; kwargs...) =
    AugLag2Workspace(model, alg, x0; kwargs...)

function optimize!(workspace::AugLag2Workspace)
    @unpack lagmodel, solution, options, convcriteria = workspace
    @unpack callback, optimizer, trace = workspace
    @unpack x0, outer_iter, iter, fcalls = workspace
    @unpack primaloptimizer, dualoptimizer = optimizer
    @unpack maxiter, primaloptions, dualoptions, quadfactor = options
    @unpack prevx, x, λ = solution

    xl, xu = getmin(lagmodel), getmax(lagmodel)

    T = eltype(x)
    ni, nj = length(λ), length(x)

    # Original model
    model = lagmodel |> getparent

    auglag = getobjective(lagmodel)

    cb =
        (tr; kwargs...) -> begin
            solution = deepcopy(solution)
            solution.prevx .= solution.x
            solution.x .= getx(lagmodel)
            solution.λ .= getλ(lagmodel)
            solution.prevf = solution.f
            solution.f = getorigobjval(lagmodel)
            solution.g .= getorigconstrval(lagmodel)
            assess_convergence!(
                solution,
                lagmodel,
                options.tol,
                convcriteria,
                true,
                workspace.iter,
            )
            callback(solution)
            return hasconverged(solution)
        end

    primaloptimizerfunction(λ) = begin
        setlinweights!(auglag, λ)
        # Primal objective to be minimized
        # Calculates the objective and its gradient
        # Optim-compatible objective
        if primaloptimizer isa MMA02 || primaloptimizer isa MMA87
            primalmodel = Model()
            addvar!(primalmodel, xl, xu)
            set_objective!(primalmodel, getprimalobjective(auglag))
            result = optimize(
                primalmodel,
                primaloptimizer,
                clamp.(getx(lagmodel), xl .+ 1e-3, xu .- 1e-3),
                options = primaloptions,
                callback = cb,
                convcriteria = KKTCriteria(),
            )
            fcalls += result.fcalls
        else
            primalobj = getoptimobj(getprimalobjective(auglag), true)
            result = Optim.optimize(
                Optim.only_fg!(primalobj),
                xl,
                xu,
                clamp.(getx(lagmodel), xl .+ 1e-3, xu .- 1e-3),
                Optim.Fminbox(primaloptimizer),
                @set(primaloptions.callback = cb),
            )
            fcalls += result.f_calls
        end
        if getx(lagmodel) != result.minimizer
            primalobj(result.minimizer)
            setx!(lagmodel, result.minimizer)
        end
        if debugging[]
            @show getx(lagmodel)
        end
        return getx(lagmodel), getorigobjval(lagmodel), getorigconstrval(lagmodel)
    end

    # Dual objective to be minimized - original objective will be maximized
    # Calculates negative the objective and its gradient
    # Optim-compatible objective
    dualobj = getoptimobj(getdualobjective(auglag, primaloptimizerfunction), false)

    # Lower and upper bounds on the dual variables
    λl = zeros(ni) .+ 1e-10
    λu = fill(Inf, ni)

    # Solve the dual problem by minimizing negative the dual objective value
    for i = 1:maxiter
        setquadweight!(lagmodel, min(getquadweight(lagmodel) * quadfactor, 1e10))
        if debugging[]
            @show getquadweight(lagmodel)
        end
        λresult = Optim.optimize(
            Optim.only_fg!(dualobj),
            λl,
            λu,
            max.(copy(λ), 1e-3),
            Optim.Fminbox(dualoptimizer),
            dualoptions, #@set(dualoptions.callback = cb),
        )
        λ .= λresult.minimizer
        if debugging[]
            @show λ
            @show getx(lagmodel)
        end
        if λ != getλ(lagmodel)
            dualobj(λ)
            setλ!(lagmodel, λ)
        end
    end
    if debugging[]
        #@show getx(lagmodel)
    end

    @pack! workspace = iter, fcalls
    solution.x .= getx(lagmodel)
    solution.λ .= getλ(lagmodel)
    solution.f = getorigobjval(lagmodel)
    solution.g .= getorigconstrval(lagmodel)
    callback(solution, update = true)

    results = GenericResult(
        optimizer,
        x0,
        solution.x,
        solution.f,
        iter,
        iter == options.maxiter,
        options.tol,
        solution.convstate,
        fcalls,
    )
    return results
end

function getoptimobj(obj, minimize = true)
    optimobj(z) = optimobj(1.0, nothing, z)
    function optimobj(F, G, z)
        if G !== nothing
            val, grad = value_gradient(obj, z)
            if minimize
                G[:] .= grad
            else
                G[:] .= .-grad
            end
            if F !== nothing
                if minimize
                    return val
                else
                    return -val
                end
            end
        end
        # No gradient necessary, just return the log joint.
        if F !== nothing
            if minimize
                return obj(z)
            else
                return -obj(z)
            end
        end
        return nothing
    end
    return optimobj
end
