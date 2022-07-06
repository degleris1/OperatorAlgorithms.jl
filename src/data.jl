
function load_case(case)
    datafile = joinpath(@__DIR__, "..", artifact"ExaData", "ExaData", case)
    
    return OPFModel(ReducedSpaceEvaluator(datafile))
end
