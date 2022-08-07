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
    A = jacobian(P, x)
    b = constraints(P, zero(x))
    x .= x - (A \ (A*x - b))

    @assert norm(A*x - b) < 1e-10
    @show objective(P, x)

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
    pp = norm(dy) / (1+norm(x))
    dd = norm(dx)  #normal_cone_distance(P, x, dx) #/ (2+norm(y))

    # Construct Newton matrix
    H = hessian(P, x)
    A = jacobian(P, x)
    K = [
        H A';
        A 0*I
    ]
    
    if cond(Matrix(K)) > 1e8
        @warn cond(Matrix(K))
    end

    # Use Newton correction
    z = K \ [dx; dy]
    dx .= z[1:num_var(P)]
    dy .= z[num_var(P)+1:end]

    @assert norm(A*dx) < 1e-10

    @show norm(dx), norm(dy)
    backtrack!(P, x, y, dx, dy)
    @show norm(dx), norm(dy)

    println("\n")

    # Update
    step!(η, x, dx)
    step!(α, y, dy)

    # Project
    #project_box!(P, x)
    
    @assert norm(A*x) < 1e-10

    # Update
    gradient!(_g, P, x)
    primal_residual!(_rp, P, x, y)
    dual_residual!(_rd, P, x, y, _g)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

function backtrack!(P, x, y, dx, dy; β=0.8, α=0.01)
    dd = norm(dual_residual(P, x, y))
    
    t = 1.0
    x̂, ŷ = x - t*dx, y - t*dy
    r̂ = dual_residual(P, x̂, ŷ)

    while norm(r̂) > (1 - α * t) * dd
        @show norm(r̂) - dd
        t = β * t
        x̂, ŷ = x - t*dx, y - t*dy
        r̂ = dual_residual(P, x̂, ŷ)
    end

    @show "Done", t, norm(r̂) - dd 

    @. dx = t*dx
    @. dy = t*dy
    return dx, dy
end

# function clip_step!(Δx, x, max_rel_step_length)
#     if norm(Δx) > max_rel_step_length*(1+norm(x))
#         Δx *= max_rel_step_length * (norm(x) / norm(Δx))
#     end
#     return Δx
# end

