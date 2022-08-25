abstract type EqualityBoxProblem end

struct StandardEqualityBoxProblem{T <: Real} <: EqualityBoxProblem
    nlp
    ω::T
    _eq_indices::AbstractArray{Int, 1}
    _ineq_indices::AbstractArray{Int, 1}
    _A::AbstractSparseMatrix{T, Int}
    _qr::SuiteSparse.SPQR.QRSparse{T, Int}
    _b::AbstractArray{T, 1}
    _xmin::AbstractArray{T, 1}
    _xmax::AbstractArray{T, 1}
    _g::AbstractArray{T, 1}
    _hs::AbstractArray{Int, 1}
end

struct BoxQuadraticProblem{
    VT <: AbstractVector{<: Real}, 
    MT <: AbstractSparseMatrix{<: Real}
} <: EqualityBoxProblem
    c_q::VT
    c_l::VT
    A::MT
    b::VT
    xmin::VT
    xmax::VT
    F_A
end

# function apply_type(P::StandardEqualityBoxProblem, mat_type, vec_type)
#     new_qr = P._qr
#     return StandardEqualityBoxProblem(
#         P.nlp,
#         P.ω,
#         P._eq_indices,
#         P._ineq_indices,
#         mat_type(P._A),
#         new_qr,
#         vec_type(P._b),
#         vec_type(P._xmin),
#         vec_type(P._xmax),
#         vec_type(P._g),
#         P._hs
#     )
# end

function EqualityBoxProblem(nlp; ω=1.0, use_qr=false)
    _eq_indices = _get_eq_indices(nlp)
    _ineq_indices = _get_ineq_indices(nlp)

    m = length(_ineq_indices)
    n = length(get_lvar(nlp))

    A = _jacobian(nlp, ω)
    b = _get_b(nlp)

    if use_qr
        @time F = qr(sparse(A'))
    else
        F = nothing
    end

    xmin = [get_lvar(nlp); get_lcon(nlp)[_ineq_indices]]
    xmax = [get_uvar(nlp); get_ucon(nlp)[_ineq_indices]]

    # Get rid of those darned fixed variables! 
    for i in 1:n
        if xmin[i] == xmax[i]
            xmin[i] = -Inf
            xmax[i] = Inf
        end
    end

    _g = similar(xmin)

    # Require diagonal Hessian
    r, c = NLPModels.hess_structure(nlp)
    @assert all(r .== c)

    return StandardEqualityBoxProblem(nlp, ω, _eq_indices, _ineq_indices, A, F, b, xmin, xmax, _g, r)
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

    x, y = zero(xmin), zero(P._b)
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
    return -constraints(P, z)
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
    A = jacobian(P, z)
    
    # ∇f + A' y
    gradient!(rd, P, z)  #  rd = ∇f
    mul!(rd, A', z.dual, 1.0, 1.0)  # rd += A' y
    return rd
end

function dual_residual(P::EqualityBoxProblem, z::PrimalDual, λmin, λmax)
    rd = similar(z.primal)
    dual_residual!(rd, P, z)

    x = z.primal
    xmin, xmax = get_box(P)

    rd += -λmin + λmax
    return rd
end

# ====
# OBJECTIVE, GRADIENT, CONSTRAINTS
# ====

function objective(P::EqualityBoxProblem, z::PrimalDual)
    u = get_true_vars(P, z.primal)
    return obj(P.nlp, u)
end

function gradient(P::EqualityBoxProblem, z::PrimalDual)
    ∇f = similar(z.primal)
    return gradient!(∇f, P, z)
end

function gradient!(∇f, P::EqualityBoxProblem, z::PrimalDual)
    x = z.primal
    n = num_true_var(P)

    # Update main variables
    u = get_true_vars(P, x)
    grad!(P.nlp, u, get_true_vars(P, ∇f))

    # Update slack variables
    ds = get_slack_vars(P, ∇f)
    ds .= 0

    ∇f .*= P.ω

    return ∇f
end

function hessian(P::EqualityBoxProblem, z::PrimalDual)
    H = Diagonal(zero(z.primal))
    return hessian!(H, P, z)
end

function hessian!(H::Diagonal, P::EqualityBoxProblem, z::PrimalDual)
    n, m = num_true_var(P), num_slack_var(P)
    x = z.primal

    # Update first block
    Hu = get_true_vars(P, H.diag)
    Hu_nz = view(Hu, P._hs)

    hess_coord!(P.nlp, get_true_vars(P, x), Hu_nz)

    # Update second block
    Hs = get_slack_vars(P, H.diag)
    Hs .= 0

    H.diag .*= P.ω

    @assert minimum(H.diag) >= 0 "Min is $(minimum(H))"

    return H
end

function constraints(P::EqualityBoxProblem, z::PrimalDual)
    hu = similar(z.dual)
    return constraints!(hu, P, z)
end

function constraints!(hu, P::EqualityBoxProblem, z::PrimalDual)
    mul!(hu, P._A, z.primal)
    @. hu .-= P._b
    return hu
end

function jacobian(P::EqualityBoxProblem, z::PrimalDual)
    return P._A
end

function jacobian_transpose_product!(Jtv, P::EqualityBoxProblem, z::PrimalDual, v)
    mul!(Jtv, transpose(P._A), v)
    return Jtv
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
    return P._xmin, P._xmax
end

function get_rhs(P::EqualityBoxProblem)
    return P._b
end

# ====
# INDEXING
# ====

function get_true_vars(P::EqualityBoxProblem, x)
    n, m = num_true_var(P), num_slack_var(P)
    return view(x, 1:n)
end

function get_slack_vars(P::EqualityBoxProblem, x)
    n, m = num_true_var(P), num_slack_var(P)
    return view(x, n+1:n+m)
end

function num_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp) + num_slack_var(P)  # m + n
end

function num_true_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp)  # n
end

function num_slack_var(P::EqualityBoxProblem)
    return length(P._ineq_indices)  # m
end

function num_con(P::EqualityBoxProblem)
    return length(P._b)  # m + k
end

function num_eq_con(P::EqualityBoxProblem)
    return length(P._eq_indices)  # k
end

# ====
# NLPMODELS HELPERS
# ====

function _jacobian(nlp, ω)
    ineq = _get_ineq_indices(nlp)
    eq = _get_eq_indices(nlp)

    n, m, k = get_nvar(nlp), length(ineq), length(eq)

    u = get_x0(nlp)

    J_u = jac(nlp, u)
    J_i = J_u[ineq, :]
    J_e = J_u[eq, :]

    inds, _ = _get_fixed_vars(nlp)
    J_z = spzeros(length(inds), n)
    J_z[:, inds] .= I(length(inds))

    return ω * [
        J_z spzeros(length(inds), m)
        J_e spzeros(k, m)
        J_i -I(m)
    ]
end

function _get_eq_indices(nlp)
    return get_jfix(nlp)
end

function _get_ineq_indices(nlp)
    return setdiff(1:get_ncon(nlp), get_jfix(nlp))
end

function _get_b(nlp)
    m = length(_get_ineq_indices(nlp))
    _, b_i = _get_fixed_vars(nlp)
    return [b_i; get_lcon(nlp)[_get_eq_indices(nlp)]; zeros(m)]
end

function _get_fixed_vars(nlp)
    inds = nlp.meta.ifix
    return inds, get_lvar(nlp)[inds]
end




