abstract type AbstractStep end

function step!(dx, rule::AbstractStep, prob::EqualityBoxProblem, x)
    error("Please implement `step!(dx, rule, prob, x)")
end