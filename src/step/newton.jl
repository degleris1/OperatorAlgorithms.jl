"""
    NewtonStep <: AbstractStep

Take a step in the direction of the negative gradient.
"""
Base.@kwdef struct NewtonStep <: AbstractStep
    solver::Symbol = :explicit
    safety::Real = 0.0
    
    use_qr::Bool = true
    num_cg_iter::Int = 10

    _u0::IdDict = IdDict()
end

# TODO: Eliminate gradient and Hessian allocations
function step!(dz, rule::NewtonStep, P::EqualityBoxProblem, z)
    (; solver, safety, use_qr, num_cg_iter) = rule

    # Compute gradient ∇L
    residual!(dz, P, z)
    @. dz.dual = -dz.dual  # We want Ax - b, not b - Ax

    # Caclulate infeasibility
    pinf = norm(dz.dual)
    dinf = norm(dz.primal)

    # Construct Hessian and Jacobian  # TODO Optimize
    H = hessian(P, z)
    @. H.diag += safety
    A = jacobian(P, z)

    # Solve
    if solver == :explicit

        solve_explict!(dz, H, A)

    elseif solver == :schur_cg

        # Load previous solution
        u0 = get!(() -> zero(dz.dual), rule._u0, dz)

        # Optionally load QR factorization
        if use_qr
            qr_factors = P._qr
        else
            qr_factors = nothing
        end

        solve_schur_cg!(dz, H, A; num_iter=num_cg_iter, qr_factors=qr_factors, u0=u0)
    
    else
        error("Support solvers are [:explicit, :schur_cg]")
    end

    return dz, (primal_infeasibility=pinf, dual_infeasibility=dinf)
end

# TODO: Optimize
function solve_explict!(dz, H, A)
    n, m = length(dz.primal), length(dz.dual)

    K = [
        H A';
        A 0*I
    ]
    b = [dz.primal; dz.dual]

    # Solve Newton system
    Δ = K \ [dz.primal; dz.dual]
    #@show extrema(svdvals(Matrix(K)))

    # Update
    @. dz.primal = -Δ[1:n]
    @. dz.dual = -Δ[n+1:n+m]

    return dz
end

# TODO: Optimize
function solve_schur_cg!(dz, H, A; num_iter=10, qr_factors=nothing, u0=nothing)
    @assert all(
        H.diag .> sqrt(eps())
    ) "Hessian nearly singular: index $(argmin(H.diag)), value $(minimum(H.diag))"

    dx, dy = dz.primal, dz.dual
    u0 = something(u0, zero(dy))

    # Create right hand size
    b1 = copy(dx)
    b2 = copy(dy)

    # Eliminate upper block
    b̃ = A * (H \ b1) - b2

    # Solve subsystem without QR
    if isnothing(qr_factors)
        K = x -> A * (H \ (A' * x))
        dy .= solve_cg!(u0, K, b̃, num_iter)

    # Solve subsystem with QR
    else
        Q, R = qr_factors

        b̂ = R' \ b̃
        K = x -> Q' * (H \ (Q * x))

        u = solve_cg!(u0, K, b̂, num_iter)
        dy .= R \ u
    end

    # Back solve
    dx .= H \ (b1 - A' * dy)

    # Check error
    # @show norm(A * inv(H) * A' * dy - b̃)
    # @show norm(H * dx + A' * dy - b1)
    # @show norm(A * dx - b2)


    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual

    return dz
end

# TODO: Cache r, p, w
function solve_cg!(u0, K, b, num_iter; ϵ=1e-8, upper_tol=1e-1)
    u = u0  # We override u0
    r = b - K(u0)
    
    p = copy(r)
    w = zero(b)

    rho = [norm(r)^2]
    for iter in 1:num_iter
        if sqrt(rho[iter]) < ϵ
            break
        end

        w .= K(p)
        α = rho[iter] / (p'w)
        
        @. u = u + α*p
        @. r = r - α*w
        
        push!(rho, norm(r)^2)
        
        @. p = r + (rho[iter+1] / rho[iter]) * p
    end

    if sqrt(rho[end]) > upper_tol
        @warn "CG system only solved to: $(sqrt(rho[end])) accuracy"
    end

    return u
end
