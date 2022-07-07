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

function step!(alg::Dommel, nlp, u, λ)
    (; η, ρ, max_rel_step_length, update_dual) = alg

    # Get gradient of objective and of quadratic constraint penalty
    # The quadratic penalty is (1/2) * || diag(sqrt(s)) * h(u)_+ ||_2^2
    # Which has gradient Jh(u)_+^T * diag(s) * h(u)_+
    # Where s is a vector of weights (by default, s = ρ I)

    ∇f = gradient(nlp, u)
    h = inequality_constraints(nlp, u)
    Jh = inequality_jacobian(nlp, u)

    # Account for inactive constraints
    inactive_set = (h .< 0)
    active_set = (h .>= 0)
    hp = relu(h)
    Jh[inactive_set, :] .= 0

    # Adjust step length if needed
    Δu = η * (∇f + ρ * Jh' * hp)
    if norm(Δu) > max_rel_step_length*norm(u)
        Δu *= max_rel_step_length * norm(u) / norm(Δu)
    end

    # Update
    @. u -= Δu

    # (Side-effect) Update dual variable
    # (This is just to accuractely keep track of the dual residual)
    # Since the Lagrangian is L(u, λ) = f(u) + λ' * h(u)
    # At an optimum, we have 0 = ∇f(u) + Jh(u)' * λ
    # So λ = - pinv(Jh(u)') * ∇f(u)
    if update_dual
        λ[active_set] .= -pinv(Matrix(Jh[active_set, :])') * ∇f
        λ[inactive_set] .= 0
    end

    # Project
    project!(nlp, u, λ) 
end
