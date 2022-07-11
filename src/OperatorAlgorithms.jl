module OperatorAlgorithms

# Exports
export load_dc
export solve_ipopt
export EqualityBoxProblem, augment
export optimize!, History, distance
export Dommel, ExtraGrad

# Imports
# Data Loading
using Artifacts
using Argos: ReducedSpaceEvaluator, OPFModel
using NLPModelsJuMP: MathOptNLPModel
using PowerModels: instantiate_model, DCMPPowerModel, build_opf

# Suppress all those warnings!
import PowerModels
PowerModels.Memento.config!("error")

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
include("model/eqprob.jl")
include("model/augmented.jl")

include("algorithms/interface.jl")
include("algorithms/dommel.jl")
include("algorithms/extragrad.jl")

include("utils.jl")
include("data.jl")
include("ipopt.jl")

end
