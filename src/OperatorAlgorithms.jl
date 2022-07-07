module OperatorAlgorithms

# Exports
export load_case
export optimize!, History
export Dommel

# Imports
using Artifacts
using Argos: ReducedSpaceEvaluator, OPFModel
using LinearAlgebra: norm, pinv
using NLPModels: get_x0, get_nvar, get_ncon
using NLPModels: get_lvar, get_uvar, get_lcon, get_ucon
using NLPModels: obj, grad!
using NLPModels: cons!, jac

# Code
include("utils.jl")
include("interface.jl")
include("data.jl")
include("algorithms/interface.jl")
include("algorithms/dommel.jl")

end
