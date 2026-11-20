# Adaptive subproblem tolerance for the MMA sub-solvers: the augmented
# Lagrangian's sub_tol_schedule sets the `Tolerance` each outer iteration,
# so the inner solves follow the outer loop's accuracy needs.
module NonconvexAugLagLabMMAExt

using NonconvexAugLagLab
using NonconvexMMA: MMAOptions

function NonconvexAugLagLab._set_sub_tol(alg, o::MMAOptions, tol)
    return MMAOptions(;
        maxiter = o.maxiter,
        outer_maxiter = o.outer_maxiter,
        maxinner = o.maxinner,
        tol = NonconvexAugLagLab.Tolerance(;
            x = tol, fabs = tol, frel = 0.0, kkt = tol, infeas = tol
        ),
        s_init = o.s_init,
        s_incr = o.s_incr,
        s_decr = o.s_decr,
        store_trace = o.store_trace,
        show_trace = o.show_trace,
        auto_scale = o.auto_scale,
        keep_best = o.keep_best,
        dual_options = o.dual_options,
        convcriteria = o.convcriteria,
        verbose = o.verbose,
    )
end

end
