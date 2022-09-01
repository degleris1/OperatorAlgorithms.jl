
function BoxQP(opf; use_qr=true, block_size=16, rescale=false, cuda=false)
    if cuda
        return _BoxQP(opf; use_qr=use_qr, block_size=block_size, rescale=rescale,
            i=Int32, dv=CuVector, sv=CuSparseVector, sm=CuSparseMatrixCSC)
    else
        return _BoxQP(opf; use_qr=use_qr, block_size=block_size, rescale=rescale)
    end
end


function _BoxQP(
    nlp; 
    use_qr=false, 
    block_size=64,
    rescale=false,
    t=Float64,
    i=Int,
    dv=Vector,
    sv=SparseVector,
    sm=SparseMatrixCSC,
)
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

    if rescale
        # Set xmin := 0 (and set x̃ = x - xmin), or x = x̃ + xmin
        # x <= xmax     --> x̃ <= xmax - xmin
        # x >= xmin     --> x̃ >= 0
        # Ax = b        --> Ax̃ = b - A*xmin
        # x'Px + c'x 

        bb = (xmin .!= -Inf)
        δ = xmin[bb]

        xmin[bb] .= 0
        xmax[bb] .-= δ
        b .-= A[:, bb] * δ
        c_0 += (1/2) * δ' * (c_q[bb] .* δ) - c_l[bb]' * δ
        c_l[bb] .-= δ .* c_q[bb]

        # Scale everything so that xmax - xmin >= 1 (set x̃ = x ./ xmax), or x = Diagonal(xmax) * x̃
        bu = (xmax .!= Inf)
        δ = xmax[bu]

        xmax[bu] ./= δ
        A[:, bu] .= A[:, bu] * Diagonal(δ)
        c_l[bu] .*= δ
        c_q[bu] .*= δ .^ 2
    end

    # Factorize equality constraint matrix
    @time if use_qr
        _F = qr(sparse(A'))
        Q = BlockyHouseholderQ{t, sm{t, i}, dv{t}}(_F.Q, block_size)
        block_F = FancyQR{t, i, sm{t, i}, dv{t}, dv{i}}(
            Q, _F.R, sparse(_F.R'), _F.prow, _F.pcol, invperm(_F.prow), invperm(_F.pcol)
        )
    else
        block_F = nothing
    end

    return BoxQuadraticProblem{t, i, dv{t}, dv{i}, sm{t, i}}(c_q, c_l, c_0, A, b, xmin, xmax, block_F)
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
