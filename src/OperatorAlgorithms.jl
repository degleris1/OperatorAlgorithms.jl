module OperatorAlgorithms


# Exports
export load_case




# Imports
using Artifacts

using Argos: ReducedSpaceEvaluator, OPFModel

using NLPModels: get_x0, get_nvar, get_ncon
using NLPModels: get_lvar, get_uvar, get_lcon, get_ucon
using NLPModels: obj, grad
using NLPModels: cons, jac





# Code
include("interface.jl")
include("data.jl")

end
