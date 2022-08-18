module OperatorAlgorithms

# Exports
## Helpers
export load_dc, write_mps_dc, load_toy
export solve_ipopt

## Model
export PrimalDual
export EqualityBoxProblem, BarrierProblem

## Algorithm
export descent!
export History, distance
export Backtracking
export FixedStop
export GradientStep, NewtonStep

#export precondition_cp, precondition_ruiz, precondition_cp_ruiz
#export optimize!, History, distance
#export FixedStep, TrustStep, AdaptiveStep, get_good_step
#export RMSProp, Momentum, RAdam, AMSGrad
#export Dommel, HybridGradient, Restarted, Continuation, Newton, TruncNewton

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
import NLPModels
MOI = MathOptInterface

# Utilities
using SparseArrays: spzeros, sparse
using LinearAlgebra: pinv, I, opnorm, Diagonal, cond, svdvals, mul!, diag, qr
import LinearAlgebra: norm

# Modeling
using NLPModels: get_x0, get_nvar, get_ncon
using NLPModels: get_lvar, get_uvar, get_lcon, get_ucon, get_jfix
using NLPModels: obj, grad!, hess, hess_coord!
using NLPModels: cons!, jac, jtprod!

# Solvers
using NLPModelsIpopt: ipopt

# Code
## Primal dual object
include("primal_dual_vector.jl")

## Optimization problems
include("model/equality_prob.jl")
include("model/barrier.jl")

## Algorithm core code
include("alg/history.jl")
include("alg/stepsize.jl")
include("alg/converged.jl")
include("step/interface.jl")
include("alg/optimizer.jl")

## Step rules
include("step/gradient.jl")
include("step/newton.jl")

## Utilities
include("utils.jl")
include("data.jl")
include("ipopt.jl")

end
