abstract type EqualityBoxProblem end

struct BoxQuadraticProblem{
    T <: Real,
    I <: Integer,
    DV <: AbstractVector{T},
    DVI <: AbstractVector{I},
    SM <: AbstractSparseMatrix{T, I},
} <: EqualityBoxProblem
    c_q::DV
    c_l::DV
    c_0::Real
    A::SM
    b::DV
    xmin::DV
    xmax::DV
    F_A::FancyQR{T, I, SM, DV, DVI}
end

function initialize(P::EqualityBoxProblem; dual=false)
    xmin, xmax = deepcopy(get_box(P))

    xmin .= max.(xmin, minimum(xmin[xmin .!= -Inf]))
    xmax .= min.(xmax, maximum(xmax[xmax .!= Inf], init=maximum(xmin)+10))

    x, y, λ, μ = zero(xmin), zero(P.b), zero(xmin), zero(xmin)
    x .= (1/2) .* (xmax - xmin) .+ xmin

    @assert all(isfinite.(x))

    if dual
        λ .= 1
        μ .= 1
    end

    return PrimalDual(x, y, λ, μ)
end

# ====
# RESIDUALS
# ====

function residual(P::EqualityBoxProblem, z::PrimalDual, t=0)
    dz = zero(z)
    return residual!(dz, P, z, t)
end

function residual!(dz, P::EqualityBoxProblem, z::PrimalDual, t=0)
    primal_residual!(dz.dual, P, z)
    dual_residual!(dz.primal, P, z)
    centrality_residual!(dz.low_dual, dz.upp_dual, P, z, t)
    return dz
end

function primal_residual(P::EqualityBoxProblem, z::PrimalDual)
    r = zero(z.dual)
    return constraints!(r, P, z)
end

function primal_residual!(rp, P::EqualityBoxProblem, z::PrimalDual)
    constraints!(rp, P, z)
    @. rp = -rp
    return rp
end

function dual_residual(P::EqualityBoxProblem, z::PrimalDual)
    rd = zero(z.primal)
    return dual_residual!(rd, P, z)
end

function dual_residual!(rd, P::EqualityBoxProblem, z::PrimalDual)
    x, y, λ, μ = z.primal, z.dual, z.low_dual, z.upp_dual
    
    # Gradient
    gradient!(rd, P, z)

    # Equality dual
    mul!(rd, transpose(P.A), y, 1, 1)

    # Inequality dual
    @. rd -= λ
    @. rd += μ

    return rd
end

function dual_residual(P::EqualityBoxProblem, z::PrimalDual, λ, μ)
    xmin, xmax = get_box(P)
    x = z.primal
    
    rd = zero(z.primal)
    
    # r = r_bar - λ + μ
    dual_residual!(rd, P, z)
    rd += -λ + μ
    
    return rd
end

function centrality_residual!(rcl, rcu, P::EqualityBoxProblem, z::PrimalDual, t)
    x, λ, μ = z.primal, z.low_dual, z.upp_dual
    xmin, xmax = get_box(P)
    
    @. rcl = -λ * (xmin - x) * isfinite(xmin) - t
    @. rcu = -μ * (x - xmax) * isfinite(xmax) - t

    return rcl, rcu
end

function centrality_residual(P::EqualityBoxProblem, z::PrimalDual, t)
    rcl, rcu = zero(z.low_dual), zero(z.upp_dual)
    centrality_residual!(rcl, rcu, P, z, t)
    return rcl, rcu
end

# ====
# OBJECTIVE, GRADIENT, CONSTRAINTS
# ====

function objective(P::EqualityBoxProblem, z::PrimalDual)
    x = z.primal

    h = P.c_q
    c = P.c_l

    return (1/2) * x' * (h .* x) + c' * x + P.c_0
end

function gradient(P::EqualityBoxProblem, z::PrimalDual)
    ∇f = similar(z.primal)
    return gradient!(∇f, P, z)
end

function gradient!(∇f, P::EqualityBoxProblem, z::PrimalDual)
    x = z.primal
    # ∇f = H*x + c
    @. ∇f = P.c_q * x + P.c_l
    return ∇f
end

function hessian(P::EqualityBoxProblem, z::PrimalDual)
    return Diagonal(hessian_diag(P, z))
end

function hessian_diag(P::EqualityBoxProblem, z::PrimalDual)
    h = similar(z.primal)
    return hessian_diag!(h, P, z)
end

function hessian_diag!(h, P::EqualityBoxProblem, z::PrimalDual)
    @. h = P.c_q
    return h
end

function constraints(P::EqualityBoxProblem, z::PrimalDual)
    hu = similar(z.dual)
    return constraints!(hu, P, z)
end

function constraints!(hu, P::EqualityBoxProblem, z::PrimalDual)
    mul!(hu, P.A, z.primal)
    @. hu -= P.b
    return hu
end

function feasible(P::EqualityBoxProblem, z::PrimalDual)
    x = z.primal
    xmin, xmax = get_box(P)
    return all(xmin .<= x .<= xmax)
end

function project_box!(P::EqualityBoxProblem, z::PrimalDual)
    x = z.primal

    xmin, xmax = get_box(P)
    @. x = clamp(x, xmin, xmax)
    return x
end

# ====
# PROBLEM DATA
# ====

function get_box(P::EqualityBoxProblem)
    return P.xmin, P.xmax
end

function get_rhs(P::EqualityBoxProblem)
    return P.b
end
