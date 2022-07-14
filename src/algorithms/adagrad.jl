"""
    Adagrad <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: primal step size
- `α = η`: dual step size
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Adagrad <: AbstractOptimizer
    max_iter::Int = 10
    η = 1.0
    α = nothing
    max_rel_step_length = 0.25
    ϵ = 1e-6

    # State
    _rd = nothing
    _rp = nothing
    _g = nothing
    
    _hx = nothing
    _hy = nothing
end

function initialize!(alg::Adagrad, P::EqualityBoxProblem)
    x, y = initialize(P), zeros(num_con(P))

    # Set dual step size
    alg.α = something(alg.α, alg.η)

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    alg._hx = zeros(length(x))
    alg._hy = zeros(length(y))

    return x, y
end

function step!(alg::Adagrad, P::EqualityBoxProblem, x, y)
    (; η, α, ϵ, max_rel_step_length, _g, _rp, _rd, _hx, _hy) = alg

    # Important:
    # At each step, we assume the residuals have already been computed
    ∇L_x = _rd
    ∇L_y = _rp

    # Get infeasibility
    pp = norm(∇L_y) / (1+norm(x))
    dd = normal_cone_distance(P, x, ∇L_x) / (1+norm(y))

    # Update diagonal Hessian approximation
    @. _hx = 0.9 * _hx + ∇L_x ^ 2
    @. _hy = 0.9 * _hy + ∇L_y ^ 2

    # Update
    @. x -= η * ∇L_x / sqrt(_hx + ϵ)
    @. y += α * ∇L_y / sqrt(_hy + ϵ)

    # Project
    project_box!(P, x)

    # Update
    gradient!(_g, P, x)
    primal_residual!(_rp, P, x, y)
    dual_residual!(_rd, P, x, y, _g)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

function clip_step!(Δx, x, max_rel_step_length)
    if norm(Δx) > max_rel_step_length*(1+norm(x))
        Δx *= max_rel_step_length * (norm(x) / norm(Δx))
    end
    return Δx
end

