"""
    HybridGradient <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: primal step size
- `α = η`: dual step size
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct HybridGradient <: AbstractOptimizer
    max_iter::Int = 10
    η::AbstractStep = FixedStep(1.0)
    α::AbstractStep = FixedStep(1.0)
    θ = 1.0

    # State
    _rd = nothing
    _rp = nothing
    _g = nothing
    _x̄ = nothing
    _rp̄ = nothing
end

function initialize!(alg::HybridGradient, P::EqualityBoxProblem, x, y)
    @assert 0 <= alg.θ <= 1

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    alg._x̄ = zeros(length(x))
    alg._rp̄ = zeros(length(y))

    return x, y
end

function step!(alg::HybridGradient, P::EqualityBoxProblem, x, y)
    (; η, α, θ, _g, _rp, _rd, _x̄, _rp̄) = alg

    # Important:
    # At each step, we assume the residuals have already been computed
    dx, dy = _rd, _rp

    # Get infeasibility
    pp = norm(dy) / (norm(x) + 1e-4)
    dd = normal_cone_distance(P, x, dx) / (norm(_g) + 1e-4)

    # Main update
    
    # (Partially) update x̄
    @. _x̄ = -x

    # Update primal variable and project
    step!(η, x, dx)
    project_box!(P, x)
   
    # Finish updating x̄
    @. _x̄ += (1+θ)*x
    primal_residual!(_rp̄, P, _x̄, y)

    # Update dual variable
    step!(α, y, _rp̄)

    # Update residuals
    gradient!(_g, P, x)
    dual_residual!(_rd, P, x, y, _g)
    primal_residual!(_rp, P, x, y)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

