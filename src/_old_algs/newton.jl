"""
    Newton <: AbstractOptimizer

Solves problem via gradient descent on f(x) + ρ * || h(u)_+ ||₂²

# Fields
"""
Base.@kwdef mutable struct Newton <: AbstractOptimizer
    max_iter::Int = 10
    η::AbstractStep = FixedStep(1.0)
    α::AbstractStep = FixedStep(1.0)

    # State
    _rd = nothing
    _rp = nothing
    _g = nothing
end

function initialize!(alg::Newton, P::EqualityBoxProblem, x, y)
    # Make x strictly feasible
    n = num_true_var(P)
    m = num_con(P)
    #A = [jacobian(P, x); [zeros(m, n) I(m)]]
    #b = [constraints(P, zero(x)); zeros(m)]
    #x .= x - (A \ (A*x - b))

    #@assert norm(A*x - b) < 1e-10
    #@show minimum(x)
    #@show objective(P, x)

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    return x, y
end

function step!(alg::Newton, P::EqualityBoxProblem, x, y)
    (; η, α, _g, _rp, _rd) = alg

    # Important:
    # At each step, we assume the residuals have already been computed
    dx, dy = _rd, _rp

    # Get infeasibility
    pp = norm(dy)
    dd = norm(dx)  #normal_cone_distance(P, x, dx) #/ (2+norm(y))

    nr = sqrt(pp^2 + dd^2)

    # Construct Newton matrix
    H = hessian(P, x) + I
    A = jacobian(P, x)
    K = [
        H A';
        A 0*I
    ]
    
    if cond(Matrix(K)) > 1e10
        @warn cond(Matrix(K))
    end

    # Use Newton correction
    # We negate the dual residual since we defined it as -∇_y L(x, y)
    # If we didn't do this, then the Hessian would have -A in the second row
    # which would make it no longer symmetric
    z = K \ [dx; -dy]
    dx .= z[1:num_var(P)]
    dy .= z[num_var(P)+1:end]

    t = backtrack!(P, x, y, dx, dy)
    
    # Update
    @. x -= t * dx
    @. y -= t * dy

    #@assert norm(A*x) < 1e-10

    # Update
    gradient!(_g, P, x)
    primal_residual!(_rp, P, x, y)
    dual_residual!(_rd, P, x, y, _g)

    #@show sqrt(norm(_rp)^2 + norm(_rd)^2)
    #println()
    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

function backtrack!(P, x, y, dx, dy; β=0.8, α=0.01)
    nr = sqrt(norm(dual_residual(P, x, y))^2 + norm(primal_residual(P, x, y))^2)
    
    t = 1.0
    x̂, ŷ = x - t*dx, y - t*dy

    g = gradient(P, x̂)
    dr = dual_residual(P, x̂, ŷ)
    pr = primal_residual(P, x̂, ŷ)
    nr̂ = sqrt(norm(dr)^2 + norm(pr)^2)

    while (!feasible(P, x̂)) || (nr̂ > (1 - α * t) * nr)
        t = β * t
        
        if log10(t) < -20
            t = 0
        end
        
        @. x̂ = x - t*dx
        @. ŷ = y - t*dy

        gradient!(g, P, x̂)
        dual_residual!(dr, P, x̂, ŷ, g)
        primal_residual!(pr, P, x̂, ŷ)
        nr̂ = sqrt(norm(dr)^2 + norm(pr)^2)
    end

    return t
end

# function clip_step!(Δx, x, max_rel_step_length)
#     if norm(Δx) > max_rel_step_length*(1+norm(x))
#         Δx *= max_rel_step_length * (norm(x) / norm(Δx))
#     end
#     return Δx
# end

