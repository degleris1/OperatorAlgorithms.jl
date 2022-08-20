
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

    # Initialize
    z = something(z0, initialize(P))
    t = something(t0, get_t0(P, z))

    m = 2 * length(z.primal)

    while (t*m) > ϵ
        # Formulate a barrier problem
        Pt = BarrierProblem(P, t)

        # Solve the barrier problem with initial iterate z0
        it = something(inner_tol, min(sqrt(t*m), t*m))
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
        outer_pinf = norm(primal_residual(P, z))
        outer_dinf = norm(dual_residual(P, z, λmin, λmax))
        outer_info = (true_pinf=outer_pinf, true_dinf=outer_dinf, num_step=history.num_iter)
        
        update!(history, P, z, outer_info; should_inc=false, should_force=false)

        @show t, t*m, outer_dinf, outer_pinf

        # Update t
        t = t / μ
    end

    return z, history
end

function get_t0(P::EqualityBoxProblem, z::PrimalDual)
    return 1
end