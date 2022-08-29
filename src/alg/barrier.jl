
function barrier_method!(
    step_rule::AbstractStep,
    P::EqualityBoxProblem,
    ;
    z0=nothing,
    step_size::AbstractStepSize=Backtracking(),
    history=History(),

    t0=nothing,
    ϵ=1e-5,
    μ=10.0,
    max_inner_step=500,
    inner_tol=nothing
)
    m = sum(P.xmin .!= -Inf) + sum(P.xmax .!= Inf)

    # Initialize
    z = something(z0, initialize(P))
    t = something(t0, get_t0(P, z, m))

    while (t*m) > ϵ
        # Formulate a barrier problem
        Pt = BarrierProblem(P, t)

        # Solve the barrier problem with initial iterate z0
        it = something(inner_tol, max(ϵ, t*m / μ))
        z, history = descent!(
            step_rule, 
            Pt; 
            history=history, 
            z0=z, 
            step_size=step_size,
            stop=FixedStop(max_inner_step, it),
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
        println("t = $(round(t, sigdigits=4)); pinf = $(pinf), dinf = $(dinf)")

        # Update t
        t = t / μ
    end

    return z, history
end

function get_t0(P::EqualityBoxProblem, z::PrimalDual, m)
    #P1 = BarrierProblem(P, 1.0)
    
    #∇ϕ = gradient(P1, z)
    #∇f = gradient(P, z)

    #b = -∇f
    #A = [P.A' ∇ϕ]

    #u = A \ b
    #@show norm(A*u - b)
    #@show u[end]

    #@show m
    #@show (length(z.primal) + length(z.dual)) / m
    
    #z.dual .= u[1:end-1]

    return (length(z.primal) + length(z.dual)) / m
end