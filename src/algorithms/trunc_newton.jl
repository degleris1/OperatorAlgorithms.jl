"""
    TruncNewton <: AbstractOptimizer
"""
Base.@kwdef mutable struct TruncNewton <: AbstractOptimizer
    max_iter::Int = 10
    η::AbstractStep = FixedStep(1.0)
    α::AbstractStep = FixedStep(1.0)
    k::Int = 10

    # State
    _rd = nothing
    _rp = nothing
    _g = nothing
    _z0 = nothing
end

function initialize!(alg::TruncNewton, P::EqualityBoxProblem, x, y)
    # Make x strictly feasible
    n = num_true_var(P)
    m = num_con(P)

    # Update residual vectors
    alg._g = similar(x)
    gradient!(alg._g, P, x)

    alg._rd = similar(x)
    dual_residual!(alg._rd, P, x, y, alg._g)

    alg._rp = similar(y)
    primal_residual!(alg._rp, P, x, y)

    alg._z0 = zero([x; y])

    return x, y
end

function step!(alg::TruncNewton, P::EqualityBoxProblem, x, y)
    (; η, α, k, _g, _rp, _rd, _z0) = alg

    # Important:
    # At each step, we assume the residuals have already been computed
    dx, dy = _rd, _rp

    # Get infeasibility
    pp = norm(dy)
    dd = norm(dx)  #normal_cone_distance(P, x, dx) #/ (2+norm(y))

    nr = sqrt(pp^2 + dd^2)

    # Construct Newton matrix
    H = hessian(P, x)
    A = jacobian(P, x)
    K = [
        H A';
        A 0*I
    ]

    M = Diagonal([1 ./ (diag(H) .+ 1); ones(length(y))])

    if cond(Matrix(M*K)) > 1e8
        @show log10(cond(Matrix(M*K)))
    end

    # Use Newton correction
    # We negate the dual residual since we defined it as -∇_y L(x, y)
    # If we didn't do this, then the Hessian would have -A in the second row
    # which would make it no longer symmetric
    z = solve_cg(K, [dx; -dy], _z0, k, M)
    _z0 .= z  # Save result for next time
    
    dx .= z[1:num_var(P)]
    dy .= z[num_var(P)+1:end]

    t = backtrack!(P, x, y, dx, dy)
    
    # Update
    @. x -= t * dx
    @. y -= t * dy

    # Update
    gradient!(_g, P, x)
    primal_residual!(_rp, P, x, y)
    dual_residual!(_rd, P, x, y, _g)

    return (primal_infeasibility=pp, dual_infeasibility=dd)
end

function solve_cg(K, b, z0, num_iter, M)
    z = copy(z0)
    r = b - K*z0
    
    p = copy(r)
    d = M*r
    w = zero(b)

    ρ = [norm(r)^2]
    for iter in 1:num_iter  #num_iter
        mul!(w, K, p)  # w = K*p
        α = ρ[iter] / (p'w)
        
        @. z = z + α*p
        @. r = r - α*w
        mul!(d, M, r)  # d = M*r
        
        push!(ρ, d'r)
        
        @. p = d + (ρ[iter+1] / ρ[iter]) * p
    end

    return z
end

