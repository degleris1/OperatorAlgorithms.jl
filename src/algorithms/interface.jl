
# ====
# Abstract Optimizer Interface
# ====

abstract type AbstractOptimizer end

function initialize!(alg::AbstractOptimizer, P::EqualityBoxProblem)
    return initialize(P), zeros(num_con(P))
end

function converged(alg::AbstractOptimizer, history::History, P::EqualityBoxProblem, x, y)
    return (history.num_iter >= alg.max_iter)
end

function step!(alg::AbstractOptimizer, P::EqualityBoxProblem, x, y)
    return error("Please implement!")
end

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


