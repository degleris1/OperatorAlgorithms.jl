
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

function write_mps_dc(case; path=nothing)
    path = something(path, joinpath(@__DIR__, "..", "data", case[1:end-2] * ".mps"))

    # Load power model
    pm = instantiate_model(_get_datafile(case), DCMPPowerModel, build_opf)

    # Remove quadratic coefficients
    ldata = pm.data
    PowerModels._standardize_cost_terms!(ldata["gen"], 2, "generator")
    lpm = instantiate_model(ldata, DCMPPowerModel, build_opf)

    dst = MOI.FileFormats.Model(format=MOI.FileFormats.FORMAT_MPS)
    MOI.copy_to(dst, lpm.model)
    MOI.write_to_file(dst, path)

    return dst
end
