include("util.jl")

using CUDA
using SparseArrays
using CUDA.CUSPARSE
import CUDA.CUSPARSE: CuSparseVector, CuSparseMatrixCSC
CUDA.allowscalar(false)

CuSparseVector{T, I}(x::SparseVector) where {T, I} = CuSparseVector(x)
CuSparseMatrixCSC{T, I}(x::SparseMatrixCSC) where {T, I} = CuSparseMatrixCSC(x)
BoxQPCuda(opf; use_qr=true, block_size=16) = BoxQP(opf; use_qr=use_qr, block_size=block_size, i=Int32, dv=CuVector, sv=CuSparseVector, sm=CuSparseMatrixCSC)

using Random
Random.seed!(0)

# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000, case_ACTIVSg10k, case_ACTIVSg70k
opf = load_dc("matpower/case13659pegase.m")

# Baseline
@time stats = solve_ipopt(opf)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))

# Create problem
# P = BoxQP(opf; use_qr=true)
P = BoxQPCuda(opf; use_qr=true, block_size=1024)

# Specify inner problem solver
step = NewtonStep(safety=1.0, solver=:schur_cg, num_cg_iter=500)
#step = NewtonStep()

# Set up barrier algorithm
history = History()
@time z, history = barrier_method!(step, P; history=history, ϵ=1e-2*scale, μ=5)

pstar = OperatorAlgorithms.objective(P, z)
@show history.num_iter, sum(history.cg_iters)
@show log10(minimum(history.infeasibility))
@show log10( abs(pstar - stats.objective) / abs(stats.objective))
# plt = plot_diagnostics(history, x_opt)

# using Profile
# Profile.clear()
CUDA.@profile barrier_method!(step, P; history=history, ϵ=1e-2*scale, μ=5)
# println()

# using SparseArrays, SuiteSparse

# z = OperatorAlgorithms.initialize(P)
# F = P._qr
# R = F.R

# x = rand(length(z.primal))
# y = rand(length(z.dual))

# @time Qb = BlockyHouseholderQ(F.Q, 64)
# @show norm(Qb'*x - (F.Q'*x)[1:length(z.dual)])
# @show norm(Qb*y - F.Q*[y; zeros(length(z.primal) - length(z.dual))])

# println("--")

# @time Qb'*x
# @time Qb*y

# @time F.Q*x
# @time F.Q'*x

# println()
