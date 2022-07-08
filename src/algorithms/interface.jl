
# ====
# Optimizer History
# ====

Base.@kwdef mutable struct History
    alg = []
    primal_residual = []
    dual_residual = []
    objective::Vector{Real} = []
    variable = []
    ignore::Vector{Symbol} = []
    num_iter::Int = -1
end

function initialize!(history::History, prob, x, y)
    update!(history, prob, x, y, nothing)
end

function update!(history::History, prob, x, y, alg_info)

    KEYS = [:primal_residual, :dual_residual, :objective, :variable]
    fs = [
        primal_residual, 
        dual_residual, 
        (p, x, y) -> objective(p, x), 
        (p, x, y) -> Array(get_true_vars(p, x))
    ]

    for (k, f) in zip(KEYS, fs)
        if !(k in history.ignore)
            data = getproperty(history, k)
            push!(data, f(prob, x, y))
        end
    end

    if !isnothing(alg_info)
        push!(history.alg, alg_info)
    end
    history.num_iter += 1

    return history
end

# ====
# Abstract Optimizer Interface
# ====

abstract type AbstractOptimizer end

initialize!(alg::AbstractOptimizer, P::EqualityBoxProblem) = initialize(P), zeros(num_con(P))

converged(alg::AbstractOptimizer, history::History, P::EqualityBoxProblem, x, y) = 
    (history.num_iter >= alg.max_iter)

step!(alg::AbstractOptimizer, P::EqualityBoxProblem, x, y) = error("Please implement!")

function optimize!(alg::AbstractOptimizer, P::EqualityBoxProblem; 
    u0=nothing, λ0=nothing, history=History()
)

    # Initialize
    x, y = deepcopy(something.((u0, λ0), initialize!(alg, P)))
    initialize!(history, P, x, y)

    while !converged(alg, history, P, x, y)
        # Run one iteration of the algorithm
        alg_info = step!(alg, P, x, y)

        # Record info
        update!(history, P, x, y, alg_info)
    end

    return x, y, history, alg
end


