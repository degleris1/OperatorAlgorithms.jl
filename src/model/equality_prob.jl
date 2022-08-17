abstract type EqualityBoxProblem end

struct StandardEqualityBoxProblem{T <: Real} <: EqualityBoxProblem
    nlp
    ω
    _eq_indices
    _ineq_indices
    _A
    _qr
    _b::Vector{T}
    _xmin::Vector{T}
    _xmax::Vector{T}
    _g::Vector{T}
end

function EqualityBoxProblem(nlp, ω=1.0)
    _eq_indices = _get_eq_indices(nlp)
    _ineq_indices = _get_ineq_indices(nlp)

    _A = _jacobian(nlp, ω)
    _b = _get_b(nlp)
    Q, R = qr(_A')

    _qr = Q[:, 1:length(_b)], R

    xmin = [get_lvar(nlp); get_lcon(nlp)[_ineq_indices]]
    xmax = [get_uvar(nlp); get_ucon(nlp)[_ineq_indices]]

    _g = similar(xmin)

    # Require diagonal Hessian
    r, c = NLPModels.hess_structure(nlp)
    @assert all(r .== c)

    return StandardEqualityBoxProblem(nlp, ω, _eq_indices, _ineq_indices, _A, _qr, _b, xmin, xmax, _g)
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

    x, y = zeros(num_var(P)), zeros(num_con(P))
    x .= rand(length(x)) .* (xmax - xmin) .+ xmin
    
    @assert !any(isnan.(x))

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
    ∇f = gradient!(P._g, P, z)

    # ∇f + A' y
    jacobian_transpose_product!(rd, P, z, z.dual)
    @. rd += ∇f
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
    grad!(P.nlp, u, view(∇f, 1:n))

    # Update slack variables
    ∇f[n+1:end] .= 0

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
    hess_coord!(P.nlp, view(x, 1:n), view(H.diag, 1:n))

    # Update second block
    H.diag[n+1:n+m] .= 0

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
    return view(x, 1:num_true_var(P))
end

function get_slack_vars(P::EqualityBoxProblem, x)
    return view(x, num_true_var(P)+1:num_true_var(P)+num_slack_var(P))
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
    return get_ncon(P.nlp)  # m + k
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

    return ω * [
        J_i -I(m)
        J_e spzeros(k, m)
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
    return [get_lcon(nlp)[_get_eq_indices(nlp)]; zeros(m)]
end







