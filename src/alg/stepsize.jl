"""
    AbstractStepSize
"""
abstract type AbstractStepSize end

"""
    adjust_step!(dx, rule::AbstractStepSize, prob::EqualityBoxProblem, x)

Finds a scalar `t` to multiply `dx` by. Potentially modifies `dx` in other ways. Returns `t`
and does NOT scale `dx` by `t`.
"""
function adjust_step!(dx, rule::AbstractStepSize, prob::EqualityBoxProblem, x)
    error("Please implement.")
end

"""
    Backtracking <: AbstractStepSize

Use a backtracking line search to determine the step size.
"""
mutable struct Backtracking <: AbstractStepSize
    α::Real
    β::Real
    xhat
    rhat
end

Backtracking(; α=0.01, β=0.8) = Backtracking(α, β, IdDict(), IdDict())

function adjust_step!(dx, R::Backtracking, P::EqualityBoxProblem, x)
    (; α, β, xhat, rhat) = R

    x̂ = get!(() -> zero(x), xhat, x)
    r̂ = get!(() -> zero(x), rhat, x)

    nr = norm(residual!(r̂, P, x))  # Initial residual
    t = 1.0

    update!(x̂, x, t, dx)
    nr_hat = feasible(P, x̂) ? norm(residual!(r̂, P, x̂)) : Inf

    while nr_hat > (1 - α * t) * nr
        t = β * t

        # Step rejected
        if t < 1e-20
            t = 0
        end

        update!(x̂, x, t, dx)
        nr_hat = feasible(P, x̂) ? norm(residual!(r̂, P, x̂)) : Inf
    end

    return t
end
