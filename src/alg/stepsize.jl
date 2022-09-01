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

    t_max = max_step_in_box(P, x, dx)
    @show t_max

    residual!(r̂, P, x)
    nr = norm(r̂)  # Initial residual

    # Set first x̂
    t = min(1.0, t_max - sqrt(eps()))
    update!(x̂, x, t, dx)

    if feasible(P, x̂)
        residual!(r̂, P, x̂)
        nr_hat = norm(r̂)
    else
        nr_hat = Inf
    end

    while nr_hat > (1 - α * t) * nr
        # @show nr_hat - nr
        t = β * t

        # Step rejected
        if t < 1e-20
            t = 0
            break
        end

        update!(x̂, x, t, dx)

        # Update residual
        if feasible(P, x̂)
            residual!(r̂, P, x̂)
            nr_hat = norm(r̂)
        else
            nr_hat = Inf
        end
    end

    @show nr_hat

    return t
end

function max_step_in_box(P, z, dz)
    x = z.primal
    dx = dz.primal
    xmin, xmax = get_box(P)

    Δ_low = x - xmin
    t_low = minimum(Δ_low ./ max.(0, -dx))

    Δ_upp = xmax - x
    t_upp = minimum(Δ_upp ./ max.(0, dx))

    return min(t_low, t_upp)
end
