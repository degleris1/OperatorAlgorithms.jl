module OperatorAlgorithms

# Exports
export load_dc, write_mps_dc, load_toy
export solve_ipopt
export EqualityBoxProblem, BarrierProblem, augment
export precondition_cp, precondition_ruiz, precondition_cp_ruiz
export optimize!, History, distance
export FixedStep, TrustStep, AdaptiveStep, get_good_step
export RMSProp, Momentum, RAdam, AMSGrad
export Dommel, HybridGradient, Restarted, Continuation, Newton, TruncNewton

# Imports
# Data Loading
using LazyArtifacts
using Argos: ReducedSpaceEvaluator, OPFModel
using NLPModelsJuMP: MathOptNLPModel
using PowerModels: instantiate_model, DCMPPowerModel, build_opf
using JuMP: @objective, @variable, @constraint, @NLobjective

import MathOptInterface
import PowerModels
import JuMP
MOI = MathOptInterface

# Utilities
using LinearAlgebra: norm, pinv, I, opnorm, Diagonal, cond, svdvals, mul!, diag, qr

# Modeling
using NLPModels: get_x0, get_nvar, get_ncon
using NLPModels: get_lvar, get_uvar, get_lcon, get_ucon, get_jfix
using NLPModels: obj, grad!, hess
using NLPModels: cons!, jac, jtprod!

# Solvers
using NLPModelsIpopt: ipopt

# Code
include("model/equality_prob.jl")
include("model/barrier.jl")
include("model/augmented.jl")
include("model/precond.jl")
include("model/regularize.jl")

include("algorithms/history.jl")
include("algorithms/step.jl")
include("algorithms/interface.jl")

include("algorithms/dommel.jl")
include("algorithms/cp.jl")
include("algorithms/newton.jl")
include("algorithms/trunc_newton.jl")
include("algorithms/restart.jl")
include("algorithms/continuation.jl")
# include("algorithms/extragrad.jl")

include("utils.jl")
include("data.jl")
include("ipopt.jl")

end
