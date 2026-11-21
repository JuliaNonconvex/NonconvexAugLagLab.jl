# Adaptive subproblem tolerance for the Uno sub-solver: the augmented
# Lagrangian's sub_tol_schedule sets Uno's primal and dual tolerances each
# outer iteration, so the inner solves follow the outer loop's accuracy needs
# (loose early, tight late) instead of a fixed tolerance.
module NonconvexAugLagLabUnoExt

using NonconvexAugLagLab
using NonconvexUno: UnoOptions

function NonconvexAugLagLab._set_sub_tol(alg, o::UnoOptions, tol)
    return UnoOptions(
        (; o.nt..., primal_tolerance=tol, dual_tolerance=tol), o.linear_constraints
    )
end

end
