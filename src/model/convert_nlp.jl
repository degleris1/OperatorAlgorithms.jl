
function BoxQP(nlp; use_qr=false)
    eq_indices = _get_eq_indices(nlp)
    ineq_indices = _get_ineq_indices(nlp)

    m = length(ineq_indices)
    n = length(get_lvar(nlp))
    u = get_x0(nlp)

    # Construct objective
    H = NLPModels.hess(nlp, u)
    ∇f = NLPModels.grad(nlp, u)
    _c = ∇f - H*u
    
    @assert Diagonal(H) == H  # Must be diagonal! 

    c_q = sparse([diag(H); zeros(m)])
    c_l = sparse([_c; zeros(m)])
    c_0 = obj(nlp, u) - (1/2)*dot(u, H, u) - _c'u

    # Construct equality constraints
    A = _jacobian(nlp)
    b = _get_b(nlp)

    # Construct lower and upper bounds
    xmin = [get_lvar(nlp); get_lcon(nlp)[ineq_indices]]
    xmax = [get_uvar(nlp); get_ucon(nlp)[ineq_indices]]

    # Get rid of those darned fixed variables! 
    for i in 1:n
        if xmin[i] == xmax[i]
            xmin[i] = -Inf
            xmax[i] = Inf
        end
    end

    # Factorize equality constraint matrix
    @time F = use_qr ? qr(sparse(A')) : nothing
    return BoxQuadraticProblem(c_q, c_l, c_0, A, b, xmin, xmax, F)
end

function _jacobian(nlp)
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

    return [
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
