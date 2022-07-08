
function _get_datafile(case)
    return joinpath(@__DIR__, "..", artifact"ExaData", "ExaData", case)
end

function load_ac_reduced(case)
    return OPFModel(ReducedSpaceEvaluator(_get_datafile(case)))
end

function load_dc(case)
    pm = instantiate_model(_get_datafile(case), DCMPPowerModel, build_opf)
    return MathOptNLPModel(pm.model)
end

