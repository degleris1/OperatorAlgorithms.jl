mutable struct AugmentedEqualityBoxProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    ρ1::Real
    ρ2::Real
    _h
    _Jth
end

function Base.getproperty(P::AugmentedEqualityBoxProblem, s::Symbol)
    if s === :nlp
        return P.P.nlp
    else
        return getfield(P, s)
    end
end

function augment(P::StandardEqualityBoxProblem, ρ) 
    if ρ > 1
        ρ1, ρ2 = 1/sqrt(ρ), sqrt(ρ)
    else
        ρ1, ρ2 = 1, ρ
    end

    h = zeros(num_con(P))
    Jth = zeros(num_var(P))

    return AugmentedEqualityBoxProblem(P, 1/ρ1, ρ2, h, Jth)
end

function gradient(P::AugmentedEqualityBoxProblem, x)
    ∇f = gradient(P.P, x)  # Use gradient of sub-problem
    Jh = jacobian(P, x)
    h = constraints(P, x)

    (; ρ1, ρ2) = P
    return ρ1 * ∇f + ρ2 * Jh' * h
end

function gradient!(g, P::AugmentedEqualityBoxProblem, x)
    (; ρ1, ρ2) = P
    # Compute ρ1 * ∇f + ρ2 * Jh' * h

    # ρ1 * ∇f
    gradient!(g, P.P, x)

    g .*= ρ1

    # + ρ2 * Jh' * h
    constraints!(P._h, P, x)
    jacobian_transpose_product!(P._Jth, P, x, P._h)
    P._Jth .*= ρ2

    g .+= P._Jth

    return g
end

function objective(P::AugmentedEqualityBoxProblem, x)
    (; ρ1, ρ2) = P
    return ρ1 * objective(P.P, x) + (ρ2/2) * norm(constraints(P, x))^2
end
