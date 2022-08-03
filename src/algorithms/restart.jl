"""
    Restarted <: AbstractOptimizer

# Fields
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Restarted <: AbstractOptimizer
    opt::AbstractOptimizer = Dommel()
    restart_every::Int = 100
    _x = nothing
    _y = nothing
    _iter = 0
end

function Base.getproperty(A::Restarted, s::Symbol)
    if s == :max_iter
        return A.opt.max_iter
    else
        return getfield(A, s)
    end
end

function initialize!(alg::Restarted, P::EqualityBoxProblem, x, y)
    initialize!(alg.opt, P, x, y)
    alg._x = zeros(length(x))
    alg._y = zeros(length(y))

    return x, y
end

function step!(alg::Restarted, P::EqualityBoxProblem, x, y)
    (; opt, restart_every, _x, _y, _iter) = alg

    # Update average
    @. _x += x / restart_every
    @. _y += y / restart_every

    if (_iter > 0) && (_iter % restart_every == 0)
        # Restart
        @. x = _x
        @. y = _y

        # Clear memory
        @. _x = 0
        @. _y = 0
        
        # Re-initialize the algorithm
        initialize!(opt, P, x, y)
    end

    alg._iter += 1

    return step!(opt, P, x, y)
end

