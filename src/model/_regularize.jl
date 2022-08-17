mutable struct RegularizedProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    ω::Real
end

function Base.getproperty(P::RegularizedProblem, s::Symbol)
    if s === :nlp
        return P.P.nlp
    else
        return getfield(P, s)
    end
end

function primal_residual(P::RegularizedProblem, x, y)
    return primal_residual(P.P, x, y) + P.ω * y  #-(constraints(P, x) - P.ω * y)
end

function primal_residual!(rp, P::RegularizedProblem, x, y)
    primal_residual!(rp, P.P, x, y)
    @. rp += P.ω * y
    #constraints!(rp, P, x)
    #@. rp += -P.ω * y
    #@. rp = -rp
    return rp
end

function gradient(P::RegularizedProblem, x)
    ∇f = gradient(P.P, x)
    return ∇f + P.ω * x
end

function gradient!(∇f, P::RegularizedProblem, x)
    gradient!(∇f, P.P, x)
    @. ∇f += P.ω * x
    return ∇f
end

# function dual_residual(P::RegularizedProblem, x, y)
#     ∇f = gradient(P, x)
#     Jh = jacobian(P, x)
# 
#     return ∇f + Jh' * y + P.ω * x
# end
# 
# function dual_residual!(rd, P::RegularizedProblem, x, y, g)
#     # Compute T (∇f + Jh' y)
#     
#     # First, compute Jh' * y
#     jacobian_transpose_product!(rd, P, x, y)
# 
#     # Add ∇f
#     rd .+= g
# 
#     # Add regularizer
#     @. rd += P.ω * x
# 
#     return rd
# end

