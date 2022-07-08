module OperatorAlgorithms

# Exports
export load_dc
export EqualityBoxProblem
export optimize!, History
export Dommel

# Imports
# Data Loading
using Artifacts
using Argos: ReducedSpaceEvaluator, OPFModel
using NLPModelsJuMP: MathOptNLPModel
using PowerModels: instantiate_model, DCMPPowerModel, build_opf

# Utilities
using LinearAlgebra: norm, pinv, I

# Modeling
using NLPModels: get_x0, get_nvar, get_ncon
using NLPModels: get_lvar, get_uvar, get_lcon, get_ucon
using NLPModels: obj, grad!
using NLPModels: cons!, jac

# Solvers
using NLPModelsIpopt: ipopt

# Code
include("utils.jl")
# include("interface.jl")
include("eqprob.jl")
include("data.jl")
include("ipopt.jl")
include("algorithms/interface.jl")
include("algorithms/dommel.jl")

end
