"""
    ExtraGrad <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
- `max_iter = 10`: number of iterations
- `η = 1.0`: primal step size
- `α = η`: dual step size
- `max_rel_step_length = 0.25`: trust parameter dictating the maximum step length relative to
    ||x||
"""
Base.@kwdef mutable struct ExtraGrad <: AbstractOptimizer
    max_iter::Int = 10
    η = 1.0
    α = nothing
    max_rel_step_length = 0.25
end

function step!(alg::ExtraGrad, P::EqualityBoxProblem, x, y)
    (; η, α, max_rel_step_length) = alg

    error("Deprecated")




    α = something(α, η)
    
    # Get extrapolated point
    ∇L_x = dual_residual(P, x, y)
    ∇L_y = primal_residual(P, x, y)
    
    x̃ = x - η * ∇L_x
    ỹ = y + α * ∇L_y

    # Recompute gradient at extrapolated point
    ∇L_x̃ = dual_residual(P, x̃, ỹ)
    ∇L_ỹ = primal_residual(P, x̃, ỹ)

    Δx, Δy = η * ∇L_x̃, α * ∇L_ỹ
    clip_step!(Δx, x, max_rel_step_length)
    clip_step!(Δy, y, max_rel_step_length)

    # Finally, take a step
    @. x -= Δx
    @. y += Δy

    # Project
    project_box!(P, x)

    return 0
end

