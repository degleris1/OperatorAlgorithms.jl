function PhaseOneProblem(P::BoxQuadraticProblem)
    n0 = length(P.xmin)
    m0 = length(P.b)

    n = 3 * n0
    m = m0 + 2n0

    c_q = [zero(P.c_q); zero(P.c_q); zero(P.c_q)]
    c_l = [zero(P.c_q); zero(P.c_q) .+ 1; zero(P.c_q) .+ 1]
    c_0 = 0.0

    A = [
        P.A     spzeros(m0, n0) spzeros(m0, n0)
        I(n0)   -I(n0)      0*I(n0)
        -I(n0)  0*I(n0)     I(n0)
    ]

    xmax_new = copy(P.xmax)
    xmax_new[P.xmax .== Inf] .= 0

    xmin_new = copy(P.xmin)
    xmin_new[P.xmin .== -Inf] .= 0

    b = [
        P.b;
        xmax_new;
        xmin_new
    ]

    @assert all(isfinite.(b))
    @assert all(isfinite.(A.nzval))

    s_upp = zero(P.xmax)
    s_upp[P.xmax .== Inf] .= -Inf

    s_low = zero(P.xmin)
    s_low[P.xmin .== -Inf] .= -Inf

    xmin = [
        zero(P.xmin) .- Inf;
        s_upp;
        s_low
    ]

    xmax = Inf .+ [
        zero(P.xmax);
        zero(P.xmax);
        zero(P.xmin)
    ]
    
    @show maximum(xmin)
    @show minimum(xmax)

    bh_type = typeof(P.F_A.Q)
    qr_type = typeof(P.F_A)
    p_type = typeof(P)

    _F = qr(sparse(A'))
    Q = bh_type(_F.Q, P.F_A.Q.block_size)
    block_F = qr_type(Q, _F.R, sparse(_F.R'), _F.prow, _F.pcol, invperm(_F.prow), invperm(_F.pcol))

    return p_type(c_q, c_l, c_0, A, b, xmin, xmax, block_F)
end