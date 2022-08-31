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
function step!(
    dz::PrimalDual{T}, rule::NewtonStep, P::EqualityBoxProblem, z::PrimalDual{T}
) where {T <: Real}
    (; solver, safety, use_qr, num_cg_iter) = rule

    # Compute gradient ∇L
    residual!(dz, P, z)
    @. dz.dual = -dz.dual  # We want Ax - b, not b - Ax

    # Caclulate infeasibility
    pinf = norm(dz.dual)
    dinf = norm(dz.primal)
    tinf = sqrt(pinf^2 + dinf^2)

    # Construct Hessian and Jacobian  # TODO Optimize
    A = P.A
    H = hessian(P, z)

    #@show maximum(H.diag), minimum(H.diag[H.diag .> 0])
    #@show pinf, dinf
    #println("Hessian Extrema: $(maximum(H.diag))")
    
    @. H.diag += safety

    # Report condition number
    #println("Condition Number: $(maximum(H.diag) / minimum(H.diag))")

    # Solve
    if solver == :explicit

        solve_explict!(dz, H, A)
        cnt = 1
        cg_error = 0

    elseif solver == :schur_cg

        # Load previous solution and QR factors
        u0 = get!(() -> zero(dz.dual), rule._u0, dz)
        qr_factors = use_qr ? P.F_A : nothing
        dz, cnt, cg_error = 
            solve_schur_cg!(dz, H, A; num_iter=num_cg_iter, qr_factors=qr_factors, u0=u0)
    
    else
        error("Support solvers are [:explicit, :schur_cg]")
    end

    return dz, (
        primal_infeasibility=pinf, 
        dual_infeasibility=dinf, 
        infeasibility=tinf,
        cg_iters = cnt,
        cg_error = cg_error
    )
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
