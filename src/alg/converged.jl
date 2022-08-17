abstract type AbstractStoppingCriterion end

function converged(rule::AbstractStoppingCriterion, history::History, prob::EqualityBoxProblem, x, dx)
    error("Please implement.")
end

"""
    FixedStop <: AbstractStoppingCriterion

A fixed stopping criteria that just checks whether or not the norm of the residual
is less than the tolerance (or whether the iteration limit has been reached).
"""
struct FixedStop <: AbstractStoppingCriterion
    max_iter::Int
    tol::Real
end

FixedStop(; max_iter=100, tol=1e-5) = FixedStop(max_iter, tol)

function converged(rule::FixedStop, history::History, prob::EqualityBoxProblem, x, dx)
    return (history.num_iter > rule.max_iter) || (norm(dx) <= rule.tol)
end