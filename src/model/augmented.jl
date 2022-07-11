mutable struct AugmentedEqualityBoxProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    ρ1::Real
    ρ2::Real
end

function Base.getproperty(P::AugmentedEqualityBoxProblem, s::Symbol)
    if s === :nlp
        return P.P.nlp
    else
        return getfield(P, s)
    end
end

function augment(P::StandardEqualityBoxProblem, ρ) 
    return AugmentedEqualityBoxProblem(P, 1/sqrt(ρ), sqrt(ρ))
end

function gradient(P::AugmentedEqualityBoxProblem, x)
    ∇f = gradient(P.P, x)  # Use gradient of sub-problem
    Jh = jacobian(P, x)
    h = constraints(P, x)

    (; ρ1, ρ2) = P
    return ρ1 * ∇f + ρ2 * Jh' * h
end

function objective(P::AugmentedEqualityBoxProblem, x)
    (; ρ1, ρ2) = P
    return ρ1 * objective(P.P, x) + (ρ2/2) * norm(constraints(P, x))^2
end
