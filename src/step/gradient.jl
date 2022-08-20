"""
    GradientStep <: AbstractStep

Take a step in the direction of the negative gradient.
"""
struct GradientStep <: AbstractStep end

function step!(dz, rule::GradientStep, P::EqualityBoxProblem, z)
    residual!(dz, P, z)

    pinf = norm(dz.primal)
    dinf = norm(dz.dual)
    tinf = sqrt(pinf^2 + dinf^2)
    
    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual

    return dz, (primal_infeasibility=pinf, dual_infeasibility=dinf, infeasibility=tinf)
end