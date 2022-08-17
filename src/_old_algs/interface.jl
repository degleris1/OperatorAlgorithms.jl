
# ====
# Abstract Optimizer Interface
# ====

abstract type AbstractOptimizer end

function initialize(P::EqualityBoxProblem)
    xmin, xmax = get_box(P)
    xmin[xmin .== -Inf] .= min.(-1, xmax[xmin .== -Inf] .- 1)
    xmax[xmax .== Inf] .= max.(1, xmin[xmax .== Inf] .+ 1)

    x, y = zeros(num_var(P)), zeros(num_con(P))
    x .= rand(num_var(P)) .* (xmax - xmin) .+ xmin
    
    @assert !any(isnan.(x))

    return x, y
end

function initialize!(alg::AbstractOptimizer, P::EqualityBoxProblem, x, y)
    return error()
end

function converged(alg::AbstractOptimizer, history::History, P::EqualityBoxProblem, x, y)
    return (history.num_iter >= alg.max_iter)
end

function step!(alg::AbstractOptimizer, P::EqualityBoxProblem, x, y)
    return error("Please implement!")
end

function optimize!(alg::AbstractOptimizer, P::EqualityBoxProblem; 
    x0=nothing, y0=nothing, history=History()
)

    # Initialize
    x, y = something.( deepcopy.( (x0, y0) ), initialize(P) )
    initialize!(alg, P, x, y)
    initialize!(history, P, x, y)

    while !converged(alg, history, P, x, y)
        # Run one iteration of the algorithm
        alg_info = step!(alg, P, x, y)

        # Record info
        update!(history, P, x, y, alg_info)
    end

    return x, y, history, alg
end


