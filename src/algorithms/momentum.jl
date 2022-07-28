"""
    Momentum <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: primal step size
- `α = η`: dual step size
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Momentum <: AbstractOptimizer
    max_iter::Int = 10
    η = 1.0
    α = nothing
    β = 0.99
    max_rel_step_length = 0.25

    # State
    _rd = nothing
    _rp = nothing
    _g = nothing
    _py = nothing
    _px = nothing
end

function initialize!(alg::Momentum, P::EqualityBoxProblem)
    x, y = initialize(P)
    error("Deprecated")
    # Set dual step size
    alg.α = something(alg.α, alg.η)

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    alg._px = zeros(length(x))
    alg._py = zeros(length(y))

    return x, y
end

function step!(alg::Momentum, P::EqualityBoxProblem, x, y)
    (; η, α, β, max_rel_step_length, _g, _rp, _rd, _px, _py) = alg

    # Important:
    # At each step, we assume the residuals have already been computed
    ∇L_x = _rd
    ∇L_y = _rp

    # Get infeasibility
    pp = norm(∇L_y) / (1+norm(x))
    dd = normal_cone_distance(P, x, ∇L_x) / (1+norm(y))

    # Update momentum
    @. _px = β*_px + (1-β)*∇L_x
    @. _py = β*_py + (1-β)*∇L_y

    # Update vector
    @. x -= η * _px
    @. y += α * _py

    # Project
    project_box!(P, x)

    # Update
    gradient!(_g, P, x)
    primal_residual!(_rp, P, x, y)
    dual_residual!(_rd, P, x, y, _g)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

