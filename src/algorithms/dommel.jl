"""
    Dommel <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: primal step size
- `α = η`: dual step size
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
- `update_dual = true`: whether or not to report a guess of the dual variable
"""
Base.@kwdef mutable struct Dommel <: AbstractOptimizer
    max_iter::Int = 10
    η = 1.0
    α = nothing
    max_rel_step_length = 0.25
    update_dual = true
end

function step!(alg::Dommel, P::EqualityBoxProblem, x, y)
    (; η, α, max_rel_step_length, update_dual) = alg

    α = something(α, η)

    # Get the gradient of the Lagrangian with respect to x and y
    ∇L_x = dual_residual(P, x, y)
    ∇L_y = primal_residual(P, x, y)
    
    # Adjust step length if needed
    Δx = η * ∇L_x
    clip_step!(Δx, x, max_rel_step_length)

    # Update
    @. x -= Δx

    # Update dual variable
    if update_dual
        Δy = α * ∇L_y
        clip_step!(Δy, y, max_rel_step_length)
        y .+= Δy
    end

    # Project
    project_box!(P, x)

    return 0
end

function clip_step!(Δx, x, max_rel_step_length)
    if norm(Δx) > max_rel_step_length*(1+norm(x))
        Δx *= max_rel_step_length * (norm(x) / norm(Δx))
    end
    return Δx
end

