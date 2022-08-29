# ====
# CUSTOM QR TYPE
# ====

struct FancyQR{MT <: AbstractSparseMatrix{<: Real, <: Integer}}
    Q::BlockyHouseholderQ{MT, <:Any}
    R::MT
    Rt::MT
    prow::AbstractVector{Int}
    pcol::AbstractVector{Int}
end

# ====
# FASTER LMUL FOR QRSparseQ
# ====

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
        α = LinearAlgebra.dot(h, a)
        LinearAlgebra.axpy!(τl*α, h, a)
    end
    return a
end

function LinearAlgebra.lmul!(
    adjQ::Adjoint{<:Any,<:SuiteSparse.SPQR.QRSparseQ}, 
    a::Vector
)
    Q = adjQ.parent
    if size(a, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(a))"))
    end
    for l in 1:size(Q.factors, 2)
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        α = LinearAlgebra.dot(h, a)
        LinearAlgebra.axpy!(τl'*α, h, a)
    end
    return a
end
