
function barrier_method!(
    step_rule::AbstractStep,
    P::EqualityBoxProblem,
    ;
    z0=nothing,
    step_size::AbstractStepSize=Backtracking(),
    history=History(),
    t0=nothing,
    ϵ=1e-5,
    μ=5.0,
    θ=1.5,
    max_inner_step=500,
    verbose=0,
)
    @assert 1 < θ < 2
    @assert μ > 1

    m = sum(P.xmin .!= -Inf) + sum(P.xmax .!= Inf)

    # Initialize
    z = something(z0, initialize(P))
    t = something(t0, get_t0(P, z, m, μ))

    while (t*m) > ϵ
        # Formulate a barrier problem
        Pt = BarrierProblem(P, t)

        # Solve the barrier problem with initial iterate z0
        inner_tol = max(ϵ/2, t*sqrt(m))
        z, history = descent!(
            step_rule, 
            Pt; 
            history=history, 
            z0=z, 
            step_size=step_size,
            stop=FixedStop(max_inner_step, inner_tol),
        )

        # Update true infeasibility
        λmin, λmax = get_multipliers(Pt, z)
        outer_info = (
            true_pinf=norm(primal_residual(P, z)), 
            true_dinf=norm(dual_residual(P, z, λmin, λmax)),
            num_step=history.num_iter, 
            objective=objective(P, z)
        )
        
        update!(history, P, z, outer_info; should_inc=false, should_force=false)

        pinf = round(outer_info.true_pinf; sigdigits=4)
        dinf = round(outer_info.true_dinf; sigdigits=4)
        (verbose >= 1) && println("t = $(round(t, sigdigits=4)); pinf = $(pinf), dinf = $(dinf)")

        # Update t
        t = min(t/μ, t^θ)
    end

    return z, history
end

function get_t0(P::EqualityBoxProblem, z::PrimalDual, m, μ)
    
    # P1 = BarrierProblem(P, 1.0)
    # ∇f = gradient(P, z)
    # ∇ϕ = gradient(P1, z) - ∇f

    # b = -∇f
    # A = [P.A' ∇ϕ]

    # u = A \ b
    # @show norm(b)
    # @show norm(∇ϕ)
    # @show norm(A*u - b)
    # @show norm(u)
    # @show u[end]

    # z.dual .= u[1:end-1]
    # @show norm(dual_residual(P, z))

    return μ * (length(z.primal) + length(z.dual)) / m
end
