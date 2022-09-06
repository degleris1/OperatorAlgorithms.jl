include("util.jl")

using Random
Random.seed!(0)

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 500
safety = 1.0
block_size = 1024

# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000, case_ACTIVSg10k, case_ACTIVSg70k
opf = load_dc("matpower/case300.m")

# Baseline
@time stats = solve_ipopt(opf)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))

# Create problem
P = BoxQP(opf; use_qr=true, block_size=block_size)

# Barrier method
# step = NewtonStep(safety=10.0, solver=:schur_cg, num_cg_iter=max_cg_iter)
# @time z, history = barrier_method!(step, P; ϵ=rtol*scale, μ=μ, verbose=1)
# @show history.num_iter, sum(history.cg_iters)

# Primal dual method
step = NewtonStep(primal_dual=true, safety=1.0, solver=:schur_cg, num_cg_iter=max_cg_iter)
@time z, history = descent!(step, P; stop=FixedStop(100, rtol*scale), init_dual=true)
@show history.num_iter, sum(history.cg_iters)

pstar = OperatorAlgorithms.objective(P, z)
@show log10(minimum(history.infeasibility) / scale)
@show log10( abs(pstar - stats.objective) / stats.objective )
#plt = plot_diagnostics(history, x_opt)

# using Profile; Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=1e-2*scale, μ=5)
