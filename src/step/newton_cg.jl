
# TODO: Optimize
function solve_schur_cg!(dz, H, A; num_iter=10, qr_factors=nothing, u0=nothing)
    @assert all(
        H.diag .> sqrt(eps())
    ) "Hessian nearly singular: index $(argmin(H.diag)), value $(minimum(H.diag))"

    # @show extrema(H.diag), maximum(H.diag) / minimum(H.diag)

    dx, dy = dz.primal, dz.dual
    u0 = something(u0, zero(dy))

    # Create right hand size
    b1, b2 = copy(dx), copy(dy)

    # Eliminate upper block
    b̃ = A * (H \ b1) - b2

    # Solve subsystem without QR
    if isnothing(qr_factors)
        K = x -> A * (H \ (A' * x))

        u, cnt, cg_error = solve_cg!(u0, K, b̃, num_iter)
        dy .= u

    # Solve subsystem with QR
    else
        F = qr_factors
        Rt = sparse(F.R')

        b̂ = Rt_div_b(Rt, F, b̃)  # R' \ b̃

        K = x -> Qtx(F, (H \ Qx(F, x)))  # K = Q' H Q x
        u, cnt, cg_error = solve_cg!(u0, K, b̂, num_iter)

        dy .= R_div_b(F, u)  # R \ u

    end

    if sqrt(cg_error) > 1e-1
        @warn "CG system only solved to: $(sqrt(cg_error)) accuracy"
    end

    # Back solve
    dx .= H \ (b1 - A' * dy)

    # Check error
    # @show norm(A * inv(H) * A' * dy - b̃)
    # @show norm(H * dx + A' * dy - b1)
    # @show norm(A * dx - b2)

    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual

    return dz, cnt, cg_error
end

# TODO: Cache r, p, w
function solve_cg!(u0, K, b, num_iter; ϵ=1e-10, upper_tol=1e-1)
    u = u0  # We override u0
    r = b - K(u0)
    
    p = copy(r)
    w = zero(b)

    rho = [norm(r)^2]
    cnt = 0
    for iter in 1:num_iter
        if sqrt(rho[iter]) < ϵ
            break
        end

        cnt += 1

        w .= K(p)
        α = rho[iter] / (p'w)
        
        @. u = u + α*p
        @. r = r - α*w
        
        push!(rho, norm(r)^2)
        
        @. p = r + (rho[iter+1] / rho[iter]) * p
    end

    return u, cnt, sqrt(rho[end])
end

# ====
# SPARSE QR OPERATIONS
# (Since Julia sparse linear algebra is incomplete...)
# ====

function Rx(F, x)
    return F.R * x[F.pcol]
end

function Rtx(F, x)
    return (F.R' * x)[invperm(F.pcol)]
end

function Qx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return (F.Q * [x; zeros(m - n)])[invperm(F.prow)]
end

function Qtx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return (F.Q' * x[F.prow])[1:n]
end

function R_div_b(F, b)
    return (F.R \ b)[invperm(F.pcol)]
end

function Rt_div_b(F, b)
    return F.R' \ (b[F.pcol])
end

function Rt_div_b(Rt, F, b)
    return Rt \ (b[F.pcol])
end

function LinearAlgebra.lmul!(
    Q::SuiteSparse.SPQR.QRSparseQ, 
    a::Vector
)
    if size(a, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(a))"))
    end
    for l in size(Q.factors, 2):-1:1
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        α = LinearAlgebra.dot(h, a)  # OPTIMIZE (50% of runtime)
        LinearAlgebra.axpy!(τl*α, h, a)  # OPTIMIZE (25% of runtime)
    end
    return a
end

function LinearAlgebra.lmul!(
    adjQ::LinearAlgebra.Adjoint{<:Any,<:SuiteSparse.SPQR.QRSparseQ}, 
    a::Vector
)
    Q = adjQ.parent
    if size(a, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(a))"))
    end
    for l in 1:size(Q.factors, 2)
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        α = LinearAlgebra.dot(h, a)  # OPTIMIZE
        LinearAlgebra.axpy!(τl'*α, h, a)  # OPTIMIZE
    end
    return a
end



function _test_my_lmul!(Q::SuiteSparse.SPQR.QRSparseQ, A; block=1, block_size=1024)
    if size(A, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    At = sparse(A')

    for l in ((block-1)*block_size+1) : min((block * block_size), size(Q.factors, 2))
        # A = A + τl h (A'h)'
        
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)

        for j in 1:size(A, 2)
            a = view(A, :, j)
            LinearAlgebra.axpy!(τl*LinearAlgebra.dot(h, a), h, a)
        end
    end
    return A
end

function get_block(Q::SuiteSparse.SPQR.QRSparseQ; block_id=1, block_size=512)
    n = size(Q.factors, 1)
    K = size(Q.factors, 2)
    r = block_size

    block = (r*(block_id-1)+1) : min(r*block_id, K)
    V = Q.factors[:, block]
    β = Q.τ[block]

    W = spzeros(n, r)
    Y = spzeros(n, r)

    Y[:, 1] = V[:, 1]
    W[:, 1] = -β[1] * V[:, 1]

    for j in 2:r
        v = V[:, 2]
        z = -β[j]*v - β[j]*(W*(Y' * v))
        W[:, j] .= z
        Y[:, j] .= v
    end

    return W, Y
end

function get_all_blocks(Q::SuiteSparse.SPQR.QRSparseQ; block_size=512)
    K = size(Q.factors, 2)
    num_blocks = ceil(Int, K/block_size)

    return [get_block(Q; block_id=k, block_size=block_size) for k in 1:num_blocks]
end