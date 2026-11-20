# Adaptive subproblem tolerance for the NLopt sub-solvers: the augmented
# Lagrangian's sub_tol_schedule sets the absolute tolerances each outer
# iteration, so the inner solves follow the outer loop's accuracy needs.
module NonconvexAugLagLabNLoptExt

using NonconvexAugLagLab
using NonconvexNLopt: NLoptOptions

function NonconvexAugLagLab._set_sub_tol(alg, o::NLoptOptions, tol)
    return NLoptOptions((; o.nt..., ftol_abs = tol, xtol_abs = tol))
end

end
