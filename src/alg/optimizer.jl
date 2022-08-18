
function descent!(
    step_rule::AbstractStep,
    P::EqualityBoxProblem,
    ;
    z0=nothing,
    step_size::AbstractStepSize=Backtracking(),
    stop::AbstractStoppingCriterion=FixedStop(),
    history=History(),
)
    # Initialize
    started = false
    z = something(z0, initialize(P))
    dz = zero(z)

    while (!started) || (!converged(stop, history, P, z, dz))
        started = true
        
        # Choose descent direction and update history
        dz, alg_info = step!(dz, step_rule, P, z)
        update!(history, P, z, alg_info)

        # Choose step size
        t = adjust_step!(dz, step_size, P, z)

        # Update
        update!(z, z, t, dz)  # x := x + t*dx
    end

    # Update history one more time
    dz, alg_info = step!(dz, step_rule, P, z)
    update!(history, P, z, alg_info)

    return z, history
end