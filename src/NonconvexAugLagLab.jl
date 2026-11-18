module NonconvexAugLagLab

export AugLagAlg, AugLagOptions, RelaxConstraintSelection, AugLagResult
export QuasiNewtonAlg, AbstractHessianApprox
export LBFGSApprox, DenseBFGSApprox, StructuredBFGSApprox, NewtonTRApprox
export LimitedMemorySR1Approx
export IdentityPreconditioner, DiagonalPreconditioner
export SQSDPAlg, SQSDPOptions, SQSDPResult

using Reexport, Parameters, LinearAlgebra, SparseArrays
@reexport using NonconvexCore
using NonconvexCore:
    VecModel,
    AbstractOptimizer,
    AbstractModel,
    Objective,
    VectorOfFunctions,
    AbstractResult,
    GenericResult,
    Tolerance
import NonconvexCore: Workspace, optimize!, _optimize_precheck

include("qn.jl")
include("auglag.jl")
include("sqsdp.jl")

end
