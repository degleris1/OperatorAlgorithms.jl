include("util.jl")

using Random
Random.seed!(0)

# case118, case300
# case1354pegase, case2869pegase
# case_ACTIVSg2000
opf = load_dc("case300.m"; make_linear=false)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution
n = length(x_opt)

# Create problem
P = EqualityBoxProblem(opf; use_qr=true)

# Specify inner problem solver
#step = NewtonStep(safety=1.0, solver=:schur_cg, num_cg_iter=200)
step = NewtonStep()

# Set up barrier algorithm
history = History(force=[:variable])
@time z, history = barrier_method!(step, P; history=history, ϵ=1e-4 * sqrt(n), μ=5)

@show length(history.infeasibility)
@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; xscale=:identity)

# using Profile
# Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=15.0, μ=10);