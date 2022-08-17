"""
    Continuation <: AbstractOptimizer

# Fields
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Continuation <: AbstractOptimizer
    opt::AbstractOptimizer = Dommel()
    _weight::Real = 1.0

    P0 = nothing
    _rp = nothing
    _rd = nothing
    _g = nothing
end

function Base.getproperty(A::Continuation, s::Symbol)
    if s == :max_iter
        return A.opt.max_iter
    else
        return getfield(A, s)
    end
end

function initialize!(alg::Continuation, P::EqualityBoxProblem, x, y)
    alg.P0 = RegularizedProblem(P, alg._weight)
    initialize!(alg.opt, alg.P0, x, y) 

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    return x, y
end

function step!(alg::Continuation, P::EqualityBoxProblem, x, y)
    thresh = 10.0
    decay = 0.5

    dx = alg._rd
    dy = alg._rp
    _g = alg._g

    # Get (true) infeasibility
    pp = norm(dy) / (norm(x) + 1e-4)
    dd = normal_cone_distance(P, x, dx) / (norm(_g) + 1e-4)
    
    # Take a step
    data = step!(alg.opt, alg.P0, x, y)

    # (Optionally) update sub problem
    # if (fake) infeasibility is sufficiently low
    # And re-initialize algorithm
    err = data.primal_infeasibility^2 + data.dual_infeasibility^2
    if sqrt(err) <= thresh * alg._weight
        alg._weight *= decay
        alg.P0 = RegularizedProblem(P, alg._weight)
        initialize!(alg.opt, alg.P0, x, y) 
    end
    
    # Update residual vectors
    gradient!(alg._g, P, x)
    dual_residual!(alg._rd, P, x, y, alg._g)
    primal_residual!(alg._rp, P, x, y)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

