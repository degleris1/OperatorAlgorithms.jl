"""
"""
mutable struct QrcgOptions <: MadNLP.AbstractOption
    safety
    max_cg_iter
end

"""
"""
mutable struct QrcgSolver <: MadNLP.AbstractLinearSolver
    opt::QrcgOptions
    factors
    v
end

QrcgSolver(; safety=1.0, max_cg_iter=500) = QrcgSolver(safety, max_cg_iter, nothing, nothing)

function MadNLP.introduce(::QrcgSolver)
    println("QR-Conjugate-Gradient solver")
end

function MadNLP.factorize!(S::QrcgSolver)
    error()
end

function MadNLP.solve!(S::QrcgSolver, x::AbstractVector)
    error()
end

function is_inertia(S::QrcgSolver)
    return false
end

function inertia(S::QrcgSolver)
    error()
end



