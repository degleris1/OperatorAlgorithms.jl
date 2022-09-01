include("util.jl")

using Random
Random.seed!(0)

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 500
safety = 10.0
block_size = 1024

# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000, case_ACTIVSg10k, case_ACTIVSg70k
opf = load_dc("matpower/case_ACTIVSg70k.m")

# Baseline
@time stats = solve_ipopt(opf)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))

# Create problem
P = BoxQP(opf; use_qr=true, block_size=block_size)

# Specify inner problem solver
step = NewtonStep(safety=safety, solver=:schur_cg, num_cg_iter=max_cg_iter)
# step = NewtonStep()

# Solve phase one problem
# @time P = PhaseOneProblem(P)
# @time z, history = barrier_method!(step, P; history=history, ϵ=rtol*scale, μ=μ, verbose=1, 
#     init_feasible=true)

# Set up barrier algorithm
history = History()
@time z, history = barrier_method!(step, P; history=history, ϵ=rtol*scale, μ=μ, verbose=1)

pstar = OperatorAlgorithms.objective(P, z)
@show history.num_iter, sum(history.cg_iters)
@show log10(minimum(history.infeasibility) / scale)
@show log10( abs(pstar - stats.objective) / stats.objective )
plt = plot_diagnostics(history, x_opt)

# using Profile; Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=1e-2*scale, μ=5)
