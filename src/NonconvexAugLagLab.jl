module NonconvexAugLagLab

export AugLag2, AugLag2Options

import Optim
using Reexport, Parameters, ChainRulesCore
using Setfield, NonconvexMMA
@reexport using NonconvexCore
using NonconvexCore: AbstractOptimizer, AbstractModel
using NonconvexCore: ConvergenceState, ConvergenceCriteria
using NonconvexCore: VecModel, Tolerance, AbstractFunction
using NonconvexCore: Solution, GenericCriteria, Trace, NoCallback
using NonconvexCore: getnineqconstraints, debugging, value_gradient
using NonconvexCore: assess_convergence!, hasconverged, GenericResult

import NonconvexCore: optimize!, Workspace, getmin, getmax, getdim
import NonconvexCore: getobjective, getresiduals
import NonconvexCore: getineqconstraints, geteqconstraints
import NonconvexCore: getobjectiveconstraints

include("aggregations.jl")
include("model.jl")
include("algorithm.jl")

end
