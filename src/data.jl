
function _get_datafile(case)
    return joinpath(@__DIR__, "..", artifact"ExaData", "ExaData", case)
end

function load_ac_reduced(case)
    return OPFModel(ReducedSpaceEvaluator(_get_datafile(case)))
end

function load_dc(case; make_linear=false)
    pm = instantiate_model(_get_datafile(case), DCMPPowerModel, build_opf)

    if make_linear
        # Remove quadratic coefficients
        ldata = pm.data
        PowerModels._standardize_cost_terms!(ldata["gen"], 2, "generator")
        pm = instantiate_model(ldata, DCMPPowerModel, build_opf)
    end

    return MathOptNLPModel(pm.model)
end

function load_toy(case::Symbol; n=20, m=3, θ=0.5, μ=1)
    model = JuMP.Model()
    
    if case == :x_squared
        @variable(model, x >= 1)
        @objective(model, Min, x^2)
    elseif case == :sum_squares
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @constraint(model, x + y == 1)
        @objective(model, Min, 2x^2 + 2y^2)
    elseif case == :linear
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @constraint(model, x + y == 1)
        @objective(model, Min, 3x + y)
    elseif case == :bad_objective
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @constraint(model, x + y == 1)
        @objective(model, Min, 1e4x + y)

    # This problem has an ill-conditioned constraint
    # Specifically, very small perturbations in `x` affect the constraint
    # How might we resolve this, in general?
    # - If the constraints are linear, we could go through and rescale the constraints
    elseif case == :bad_constraint
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @constraint(model, 1000x + y == 1)
        @objective(model, Min, 3x + y)
    elseif case == :rand_qp
        _Q = randn(n, n)
        Q = _Q'_Q
        q = randn(n)

        A = rand(m, n)

        @show cond(Q), cond(A)
        @show extrema(svdvals(Q))
        @show extrema(svdvals(A))

        @variable(model, x[1:n] >= 0)
        @objective(model, Min, θ * x' * Q * x + (1-θ) * q' * x)
        @constraint(model, A * x .== 1)

    elseif case == :barrier
        _Q = randn(n, n)
        Q = _Q'_Q
        q = randn(n)

        A = rand(m, n)

        @show cond(Q), cond(A)
        @show extrema(svdvals(Q))
        @show extrema(svdvals(A))

        @variable(model, x[1:n] >= 1e-8)
        @NLobjective(model, Min, sum(q[i]*x[i] - μ*log(x[i]) for i = 1:n))
        @constraint(model, A * x .== 1)


    elseif case == :conflict
        # Constraint conflicts with objective
        error("TODO")
    else
        error("Toy case $(string(case)) does not exist")
    end

    return MathOptNLPModel(model)
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
