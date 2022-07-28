abstract type AbstractStep end

function step!(s::AbstractStep, x, dx)
    @. x -= s.η * dx
    return x
end

"""
    FixedStep <: AbstractStep

Fixed primal and dual step sizes `α` and `η`.
"""
struct FixedStep <: AbstractStep
    η::Real
end

"""
    AdaptiveStep <: AbstractStep
"""
mutable struct AdaptiveStep <: AbstractStep
    βp::Real
    βm::Real
    η::Real
    _norm_dx::Real
end

AdaptiveStep(η; βp=1.05, βm=0.8) = AdaptiveStep(βp, βm, η, Inf)

function step!(s::AdaptiveStep, x, dx)
    # Adjust step size
    ndx = norm(dx)
    if ndx <= (1+1e-4) * s._norm_dx
        s.η *= s.βp
    else
        s.η *= s.βm
    end
    s._norm_dx = ndx

    # Update
    @. x -= s.η * dx
    return x
end

function get_good_step(P::EqualityBoxProblem; ω=1, z=0.9)
    x, y = initialize(P)
    η = z / opnorm(Array(jacobian(P, x)))

    return FixedStep(η / ω), FixedStep(η * ω)
end


"""
    TrustStep <: AbstractStep
"""
# mutable struct TrustStep <: AbstractStep
#     step::AbstractStep
#     trust::Real
# end
# 
# function step!(s::TrustStep, x, dx)
#     
#     # Shrink step if needed
#     ndx = norm(dx)
#     if ndx >= s.trust
#         @. dx *= s.trust / ndx
#     end
# 
#     # Update
#     return step!(s.step, x, dx)
# end
