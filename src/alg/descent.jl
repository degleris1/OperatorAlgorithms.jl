
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
    z = z0
    if isnothing(z)
        z = initialize(P)
    end
    dz = zero(z)

    while (!started) || (!converged(stop, history, P, z, dz))
        started = true
        
        # Choose descent direction
        dz, alg_info = step!(dz, step_rule, P, z)
        
        # Choose step size
        t = adjust_step!(dz, step_size, P, z)
        @show t
        #println()

        # Update history
        alg_info = (; step_size=t, alg_info...)
        update!(history, P, z, alg_info)

        # Update
        update!(z, z, t, dz)  # x := x + t*dx
    end

    # Update history one more time
    dz, alg_info = step!(dz, step_rule, P, z)
    update!(history, P, z, alg_info; should_inc=true)

    return z, history
end
