# Adaptive subproblem tolerance for the Ipopt sub-solver: the augmented
# Lagrangian's sub_tol_schedule sets Ipopt's `tol` each outer iteration, so
# the inner solves follow the outer loop's accuracy needs (loose early,
# tight late) instead of a fixed tolerance.
module NonconvexAugLagLabIpoptExt

using NonconvexAugLagLab
using NonconvexIpopt: IpoptOptions

function NonconvexAugLagLab._set_sub_tol(alg, o::IpoptOptions, tol)
    return IpoptOptions((; o.nt..., tol=tol))
end

end
