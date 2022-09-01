CUDA.allowscalar(false)

CuSparseVector{T, I}(x::SparseVector) where {T, I} = CuSparseVector(x)

CuSparseMatrixCSC{T, I}(x::SparseMatrixCSC) where {T, I} = CuSparseMatrixCSC(x)