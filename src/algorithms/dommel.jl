"""
    Dommel <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: step size
- `ρ = 2.0`: constraint penalty function weight
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Dommel <: AbstractOptimizer
    max_iter::Int = 10
    η = 1.0
    ρ = 2.0
    max_rel_step_length = 0.25
    update_dual = true
end

function step!(alg::Dommel, P::EqualityBoxProblem, x, y)
    (; η, ρ, max_rel_step_length, update_dual) = alg

    ∇f = gradient(P, x)
    h = constraints(P, x)
    Jh = jacobian(P, x)

    # Get gradient of objective and of quadratic constraint penalty
    # The quadratic penalty is (1/2) * ρ * || h(u) ||_2^2
    # Which has gradient ρ * Jh(u)^T * h(u)

    # We use (1/sqrt(ρ)) and sqrt(ρ) to improve stability
    if ρ > 1
        ρ1, ρ2 = (1/sqrt(ρ)), sqrt(ρ)
    else
        ρ1, ρ2 = 1, ρ
    end
    Δx = η * (ρ1 * ∇f + ρ2 * Jh' * h)
    
    # Save this for reporting information
    _Δx = deepcopy(Δx) / η
 
    # Adjust step length if needed
    if norm(Δx) > max_rel_step_length*norm(x)
        Δx *= max_rel_step_length * norm(x) / norm(Δx)
    end

    # Update
    @. x -= Δx

    # (Side-effect) Update dual variable
    # (This is just to accuractely keep track of the dual residual)
    # The 'psuedo'-dual variable should be ρ*h
    if update_dual
        y .= ρ*h
    end

    # Project
    project_box!(P, x) 

    return _Δx
end
