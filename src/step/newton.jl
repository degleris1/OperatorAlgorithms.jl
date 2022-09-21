"""
    NewtonStep <: AbstractStep

Take a step in the direction of the negative gradient.
"""
Base.@kwdef mutable struct NewtonStep <: AbstractStep
    solver::Symbol = :explicit
    safety::Real = 0.0
    
    use_qr::Bool = true
    num_cg_iter::Int = 10
    precond::Symbol = :id

    primal_dual = false
    primal_dual_weight = 5.0

    _u0::IdDict = IdDict()
    _centrality_weight = 0.0
end

# TODO: Eliminate gradient and Hessian allocations
function step!(
    dz::PrimalDual{T}, rule::NewtonStep, P::EqualityBoxProblem, z::PrimalDual{T};
    verbose=0,
) where {T <: Real}
    (; solver, safety, use_qr, num_cg_iter, primal_dual, primal_dual_weight, precond) = rule
    
    # Determine centrality weight
    if primal_dual
        rcl, rcu = centrality_residual(P, z, 0.0)
        η = dot(z.low_dual, rcl) + dot(z.upp_dual, rcu)
        rule._centrality_weight = η / (2 * primal_dual_weight * length(z.primal))
    end

    # Compute gradient ∇L
    τ = rule._centrality_weight
    residual!(dz, P, z, τ)
    @. dz.dual = -dz.dual  # We want Ax - b, not b - Ax

    # Caclulate infeasibility
    pinf = norm(dz.dual)
    dinf = norm(dz.primal)
    cinf = sqrt(norm(dz.low_dual)^2 + norm(dz.upp_dual)^2)
    tinf = sqrt(pinf^2 + dinf^2 + cinf^2)

    if (verbose >= 1)
        @show pinf, dinf, τ
    end

    if !primal_dual
        @assert cinf == 0.0
        @assert iszero(rule._centrality_weight)
    end

    # Construct Hessian and Jacobian
    A = P.A
    H = hessian(P, z)
    @. H.diag += safety

    # Reduce system and update Hessian and dual residual
    if primal_dual
        x, λ, μ = z.primal, z.low_dual, z.upp_dual
        xmin, xmax = get_box(P)

        # Update dual residual
        @. dz.primal += -dz.low_dual / (xmin - x)
        @. dz.primal += dz.upp_dual / (x - xmax)

        # Update Hessian
        @. H.diag += λ / (x - xmin) + μ / (xmax - x)
    end

    # Solve
    if solver == :explicit

        solve_explict!(dz, H, A)
        cnt = 1
        cg_error = 0

    elseif solver == :schur_cg

        # Load previous solution and QR factors
        u0 = get!(() -> zero(dz.dual), rule._u0, dz)
        qr_factors = use_qr ? P.F_A : nothing
        dz, cnt, cg_error = solve_schur_cg!(dz, H, A; 
            num_iter=num_cg_iter, qr_factors=qr_factors, u0=u0, precond=precond)
    
    else
        error("Support solvers are [:explicit, :schur_cg]")
    end

    if verbose >= 1
        @show maximum(H.diag), cnt, cg_error
        println()
    end

    # Update dλ, dμ
    if primal_dual
        x, λ, μ = z.primal, z.low_dual, z.upp_dual
        xmin, xmax = get_box(P)

        @. dz.low_dual = -(1 / (xmin - x)) * (-λ * dz.primal + dz.low_dual)
        @. dz.upp_dual = -(1 / (x - xmax)) * (μ * dz.primal + dz.upp_dual)
    end

    # Flip signs for descent
    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual
    @. dz.low_dual = -dz.low_dual
    @. dz.upp_dual = -dz.upp_dual

    return dz, (
        primal_infeasibility=pinf, 
        dual_infeasibility=dinf, 
        infeasibility=tinf,
        cg_iters = cnt,
        cg_error = cg_error
    )
end

function solve_explict!(dz, H, A)
    n, m = length(dz.primal), length(dz.dual)

    K = [
        H A';
        A 0*I
    ]
    b = [dz.primal; dz.dual]

    # Solve Newton system
    Δ = K \ [dz.primal; dz.dual]

    # Update
    @. dz.primal = Δ[1:n]
    @. dz.dual = Δ[n+1:n+m]

    return dz
end
