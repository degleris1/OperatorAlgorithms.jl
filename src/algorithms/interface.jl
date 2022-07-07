
# ====
# Optimizer History
# ====

Base.@kwdef mutable struct History
    primal_residual = []
    dual_residual = []
    objective = []
    variable = []
    ignore::Vector{Symbol} = []
    num_iter::Int = -1
end

function initialize!(history::History, nlp, u, λ)
    update!(history, nlp, u, λ)
end

function update!(history::History, nlp, u, λ)

    KEYS = [:primal_residual, :dual_residual, :objective, :variable]
    fs = [get_primal_residual, get_dual_residual, get_objective, get_var]

    for (k, f) in zip(KEYS, fs)
        if !(k in history.ignore)
            data = getproperty(history, k)
            push!(data, f(nlp, u, λ))
        end
    end

    history.num_iter += 1

    return history
end

# ====
# Abstract Optimizer Interface
# ====

abstract type AbstractOptimizer end

initialize!(alg::AbstractOptimizer, nlp) = get_x0(nlp), zeros(dim_constraints(nlp))

converged(alg::AbstractOptimizer, history::History, nlp, u, λ) = history.num_iter >= alg.max_iter

step!(alg::AbstractOptimizer, nlp, u, λ) = error("Please implement!")

function optimize!(alg::AbstractOptimizer, nlp; u0=nothing, λ0=nothing, history=History())

    # Initialize
    u, λ = deepcopy(something.((u0, λ0), initialize!(alg, nlp)))
    initialize!(history, nlp, u, λ)

    while !converged(alg, history, nlp, u, λ)
        # Run one iteration of the algorithm
        step!(alg, nlp, u, λ)

        # Record info
        update!(history, nlp, u, λ)
    end

    return u, λ, history, alg
end


