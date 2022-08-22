include("util.jl")

using Random
Random.seed!(0)

# case118, case300
# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000
opf = load_dc("case2869pegase.m")

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))

# Create problem
P = EqualityBoxProblem(opf; use_qr=true)

# Specify inner problem solver
step = NewtonStep(safety=1.0, solver=:schur_cg, num_cg_iter=200)

# Set up barrier algorithm
history = History()
@time z, history = barrier_method!(step, P; history=history, ϵ=1e-4*scale, μ=5)

@show history.num_iter, sum(history.cg_iters)
@show log10(minimum(history.infeasibility))
@show OperatorAlgorithms.objective(P, z) / stats.objective
plt = plot_diagnostics(history, x_opt)

# using Profile
# Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=1e-5*sqrt(n), μ=5)
# println()