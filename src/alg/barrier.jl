
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
)
    # This denotes the TOTAL number of Newton steps during the algorithm
    MAX_INNER_STEP = 1000

    # Initialize
    z = something(z0, initialize(P))
    t = something(t0, get_t0(P, z))

    m = 2 * length(z.primal)

    while (t*m) > ϵ
        # Formulate a barrier problem
        Pt = BarrierProblem(P, t)

        # Solve the barrier problem with initial iterate z0
        z, history = descent!(
            step_rule, 
            Pt; 
            history=history, 
            z0=z, 
            step_size=step_size,
            stop=FixedStop(MAX_INNER_STEP, t*m),
        )

        @show t, t*m, last(history.infeasibility)

        # Update t
        t = t / μ

        # TODO Update dual variables between iterations
    end

    return z, history
end

function get_t0(P::EqualityBoxProblem, z::PrimalDual)
    return 1
end