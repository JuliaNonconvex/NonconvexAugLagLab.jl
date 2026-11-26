module NonconvexAugLagLab

export AugLagAlg, AugLagOptions, RelaxConstraintSelection, AugLagResult
export SecondOrderSelection
export QuasiNewtonAlg, AbstractHessianApprox
export LBFGSApprox, DenseBFGSApprox, StructuredBFGSApprox, NewtonTRApprox
export LimitedMemorySR1Approx
export AbstractCurvatureTerm, GaussNewtonTerm, CompositeHessianApprox
export JumpGaussNewtonTerm, JumpLagrangianTerm, SplitSecantApprox
export FDGradHessianApprox
export IdentityPreconditioner, DiagonalPreconditioner
export SQSDPAlg, SQSDPOptions, SQSDPResult

using Reexport, Parameters, LinearAlgebra, SparseArrays
using ChainRulesCore: ChainRulesCore
@reexport using NonconvexCore
using NonconvexCore:
    VecModel,
    AbstractOptimizer,
    AbstractModel,
    Objective,
    VectorOfFunctions,
    AbstractResult,
    GenericResult,
    Tolerance,
    EqConstraint,
    IneqConstraint,
    SDConstraint,
    FunctionInSet,
    VariableInSet,
    getinit
import NonconvexCore: Workspace, optimize!, _optimize_precheck

include("qn.jl")
include("auglag.jl")
include("setalm.jl")
include("sqsdp.jl")

end
