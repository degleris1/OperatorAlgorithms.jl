include("util.jl")

using Random
Random.seed!(0)

# case118, case300
# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000, case_ACTIVSg10k, case_ACTIVSg70k
opf = load_dc("matpower/case2869pegase.m"; make_linear=true)

# Baseline
@time stats = solve_ipopt(opf)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))

# Create problem
P = BoxQP(opf; use_qr=true)

# Specify inner problem solver
step = NewtonStep(safety=1.0, solver=:schur_cg, num_cg_iter=200)

# Set up barrier algorithm
history = History()
@time z, history = barrier_method!(step, P; history=history, ϵ=1e-4*scale, μ=5)
pstar = OperatorAlgorithms.objective(P, z)

@show history.num_iter, sum(history.cg_iters)
@show log10(minimum(history.infeasibility))
@show log10( abs(pstar - stats.objective) / abs(stats.objective))
plt = plot_diagnostics(history, x_opt)

# using Profile
# Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=1e-4, μ=5)
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