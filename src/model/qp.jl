abstract type EqualityBoxProblem end

struct BoxQuadraticProblem <: EqualityBoxProblem
    c_q::AbstractVector{<: Real}
    c_l::AbstractVector{<: Real}
    c_0::Real
    A::AbstractSparseMatrix{<: Real}
    b::AbstractVector{<: Real}
    xmin::AbstractVector{<: Real}
    xmax::AbstractVector{<: Real}
    F_A::FancyQR{<: AbstractSparseMatrix{<: Real}}
end

function initialize(P::EqualityBoxProblem)
    xmin, xmax = deepcopy(get_box(P))

    for i in eachindex(xmin)
        if xmin[i] == -Inf && xmax[i] == Inf
            xmin[i] = -10
            xmax[i] = 10
        elseif xmin[i] == -Inf
            xmin[i] = xmax[i] - 10
        elseif xmax[i] == Inf
            xmax[i] = xmin[i] + 10
        end
    end

    x, y = zero(xmin), zero(P.b)
    x .= (1/2) .* (xmax - xmin) .+ xmin

    @show norm(P.A * x - P.b)
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

    H = Diagonal(P.c_q)
    c = P.c_l

    return (1/2) * dot(x, H, x) + c' * x + P.c_0
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

    in_box = true
    @inbounds for i in 1:length(x)
        in_box = in_box && (xmin[i] <= x[i] <= xmax[i])
    end

    return in_box
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
