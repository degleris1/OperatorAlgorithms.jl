mutable struct PreconditionedProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    σ
    τ
end

function Base.getproperty(P::PreconditionedProblem, s::Symbol)
    if s === :nlp
        return P.P.nlp
    else
        return getfield(P, s)
    end
end

function primal_residual(P::PreconditionedProblem, x, y)
    return P.σ .* constraints(P, x)
end

function primal_residual!(rp, P::PreconditionedProblem, x, y)
    
    # Return -Σ * h(x)
    constraints!(rp, P, x)
    @. rp *= -P.σ
    
    return rp
end

function dual_residual(P::PreconditionedProblem, x, y)
    ∇f = gradient(P, x)
    Jh = jacobian(P, x)

    return P.τ .* (∇f + Jh' * y)
end

function dual_residual!(rd, P::PreconditionedProblem, x, y, g)
    # Compute T (∇f + Jh' y)
    
    # First, compute Jh' * y
    jacobian_transpose_product!(rd, P, x, y)

    # Add ∇f
    rd .+= g

    # Scale by T
    @. rd *= P.τ

    return rd
end

function precondition_cp(P::EqualityBoxProblem, α=1)
    x, _ = initialize(P)
    A = jacobian(P, x)

    m, n = size(A)

    σ = [1 / norm(A[i, :], α) for i in 1:m]
    τ = [1 / norm(A[:, j], 2-α) for j in 1:n]

    return PreconditionedProblem(P, σ, τ)
end

function precondition_ruiz(P::EqualityBoxProblem, num_iter=10)
    x, _ = initialize(P)
    A0 = jacobian(P, x)

    m, n = size(A0)

    A = copy(A0)
    σ = ones(m)
    τ = ones(n)

    for iter in 1:num_iter
        σ .*= [1 / sqrt(norm(A[i, :], Inf)) for i in 1:m]
        τ .*= [1 / sqrt(norm(A[:, j], Inf)) for j in 1:n]
        A  = Diagonal(σ) * A0 * Diagonal(τ)
    end

    return PreconditionedProblem(P, σ .^ 2, τ .^ 2)
end

function precondition_cp_ruiz(P::EqualityBoxProblem; α=1, num_iter=10)
    return precondition_cp(precondition_ruiz(P, num_iter), α)
end
