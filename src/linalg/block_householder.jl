# ====
# BLOCKYQ TYPE
# ====

struct BlockyHouseholderQ{
    T <: Real,
    B <: AbstractMatrix{T}, 
    V <: AbstractVector{T}
}
    block_size::Int
    fwd_blocks::Vector{Tuple{B, B}}
    rev_blocks::Vector{Tuple{B, B}}
    n::Int  # Full dimension
    m::Int  # Reduced dimension
    u::V
    v_full::V
    v_red::V
end

function BlockyHouseholderQ{T, B, V}(Q::QRSparseQ, block_size=16) where {T, B, V}
    fwd_blocks = blockify(Q, block_size)
    rev_blocks = reverse_blockify(Q, block_size)

    n = size(Q.factors, 1)
    m = Q.n

    u = zeros(block_size)
    v_full = zeros(n)
    v_red = zeros(m)

    return BlockyHouseholderQ{T, B, V}(
        block_size, fwd_blocks, rev_blocks, n, m, u, v_full, v_red
    )
end

# ====
# BASICS
# ====

function Base.length(Q::BlockyHouseholderQ)
    return Q.n * Q.m
end

function Base.size(Q::BlockyHouseholderQ)
    return (Q.n, Q.m)
end

function Base.adjoint(Q::BlockyHouseholderQ)
    return Adjoint(Q)
end

# ====
# MULTIPLICATION OPERATOR
# ====

function Base.:*(Q::BlockyHouseholderQ, x::AbstractVector)
    y = zero(Q.v_full)
    return mul!(y, Q, x, 1, 0)
end

function Base.:*(Qt::Adjoint{<:Any, <:BlockyHouseholderQ}, x::AbstractVector)
    y = zero(Qt.parent.v_red)
    return mul!(y, Qt, x, 1, 0)
end

function LinearAlgebra.mul!(
    y::V,
    Q::BlockyHouseholderQ, 
    x::V,
    α::Real,
    β::Real  
) where {V <: AbstractVector}
    # y .= β*y + α*mul_blocks(Q.fwd_blocks, [x; zeros(Q.n - Q.m)])
        
    # Set up workspace
    _u = Q.u
    _u .= 0

    _x = Q.v_full
    _x .= 0
    _x[1:Q.m] .= x

    # Main calculation
    @. y *= β
    _x = _mul_blocks!(Q.fwd_blocks, _x, _u)
    LinearAlgebra.axpy!(α, _x, y)

    return y
end

function LinearAlgebra.mul!(
    y::V,
    adjQ::Adjoint{<:Any, <:BlockyHouseholderQ}, 
    x::V,
    α::Real,
    β::Real,
) where {V <: AbstractVector}
    # y .= β*y + α*mul_blocks(adjQ.parent.rev_blocks, x)[1:adjQ.parent.m]
    Q = adjQ.parent
    
    # Set up workspace
    _u = Q.u
    _u .= 0

    _x = Q.v_full
    _x .= x

    # Main calculation
    @. y *= β
    _x = _mul_blocks!(Q.rev_blocks, _x, _u)
    LinearAlgebra.axpy!(α, view(_x, 1:Q.m), y)
    
    return y
end

function _mul_blocks!(blocks, x, u)
    for (W, Y) in blocks
        # x .= x + W*(Y'*x)

        # u = Y' * _x
        mul!(u, Y', x)

        # _x = _x + W*u
        mul!(x, W, u, 1, 1)
    end

    return x
end

# ====
# CONSTUCTOR HELPERS
# ====

function blockify(Q::SuiteSparse.SPQR.QRSparseQ, block_size=16)
    r = block_size
    K = size(Q.factors, 2)
    num_blocks = ceil(Int, K/r)

    get_block = k -> ((k-1)*r+1):min(K, k*r)

    return [make_householder_block(Q, get_block(k), r) for k in num_blocks:-1:1]
end

function reverse_blockify(Q::SuiteSparse.SPQR.QRSparseQ, block_size=16)
    r = block_size
    K = size(Q.factors, 2)
    num_blocks = ceil(Int, K/r)

    get_block = k -> reverse(((k-1)*r+1):min(K, k*r))

    return [make_householder_block(Q, get_block(k), r) for k in 1:num_blocks]
end

"""
    make_householder_block(Q::QRSparseQ, block)

Take the Householder transforms indexed by `block`, and create a block transform that
applies the transforms in `block` from left to right.
"""
function make_householder_block(Q::SuiteSparse.SPQR.QRSparseQ, block, block_size)

    n, K = size(Q.factors)
    r = block_size

    V = deepcopy(Q.factors[:, block])
    β = deepcopy(Q.τ[block])

    # Pad block if necessary
    if size(V, 2) != block_size
        V = [V spzeros(n, r-length(block))]
        β = [β; zeros(r-length(block))]
    end

    # PERFORMANCE NOTE
    # We work in a reduced space---that is, the set of rows with nonzero indices
    # Then, at the very end, we lift up to the full space
    nz_row = sort(unique(V.rowval))
    V_low = V[nz_row, :]
    n_low = length(nz_row)

    z = zeros(n_low)

    Vs = [V_low[:, j] for j in 1:r]
    Ws = [-β[1] * Vs[1]]

    U = V_low' * V_low

    for j in 2:r        
        # z = -β[j]*(v + W*Y'*v)
        v = Vs[j]
        
        # u = Y' * v
        # u is precomputed via (V'V)[:, j]

        # z = W * u
        z .= 0
        for i in 1:(j-1)  # REDUCTION
            ui = U[i, j]
            if !iszero(ui)
                LinearAlgebra.axpy!(ui, Ws[i], z)
            end
        end

        # z = -βj * (z + v)
        LinearAlgebra.axpy!(1, v, z)
        @. z *= -β[j]

        push!(Ws, sparse(z))
    end
    
    W_low = hcat(Ws...)
    W = spzeros(n, r)
    W[nz_row, :] = W_low

    return W, V
end
