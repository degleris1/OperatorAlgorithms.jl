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

function send_to_gpu(P, dense_type, sparse_type)
    F = P.F_A
    Q = F.Q

    return BoxQuadraticProblem(
        dense_type(P.c_q),
        dense_type(P.c_l),
        dense_type(P.c_0),
        sparse_type(P.A),
        dense_type(P.b),
        dense_type(P.xmin),
        dense_type(P.xmax),
    )
end

function initialize(P::EqualityBoxProblem)
    xmin, xmax = deepcopy(get_box(P))

    xmin .= max.(xmin, minimum(xmin[xmin .!= -Inf]))
    xmax .= min.(xmax, maximum(xmax[xmax .!= Inf], init=maximum(xmin)+10))

    x, y = zero(xmin), zero(P.b)
    x .= (1/2) .* (xmax - xmin) .+ xmin

    @assert all(isfinite.(x))

    return PrimalDual(x, y)
end

# ====
# RESIDUALS
# ====

function residual(P::EqualityBoxProblem, z::PrimalDual)
    dz = similar(z)
    return residual!(dz, P, z)
end

function residual!(dz, P::EqualityBoxProblem, z::PrimalDual)
    primal_residual!(dz.dual, P, z)
    dual_residual!(dz.primal, P, z)
    return dz
end

function primal_residual(P::EqualityBoxProblem, z::PrimalDual)
    r = similar(z.dual)
    return constraints!(r, P, z)
end

function primal_residual!(rp, P::EqualityBoxProblem, z::PrimalDual)
    constraints!(rp, P, z)
    @. rp = -rp
    return rp
end

function dual_residual(P::EqualityBoxProblem, z::PrimalDual)
    rd = similar(z.primal)
    return dual_residual!(rd, P, z)
end

function dual_residual!(rd, P::EqualityBoxProblem, z::PrimalDual)
    x, y = z.primal, z.dual
    
    gradient!(rd, P, z)
    mul!(rd, transpose(P.A), y, 1, 1)

    return rd
end

function dual_residual(P::EqualityBoxProblem, z::PrimalDual, λmin, λmax)
    xmin, xmax = get_box(P)
    x = z.primal
    
    rd = similar(z.primal)
    
    # r = r_barr - λmin + λmax
    dual_residual!(rd, P, z)
    rd += -λmin + λmax
    
    return rd
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
