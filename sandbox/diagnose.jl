include("util.jl")

using Random
Random.seed!(0)

opf = load_dc("case300.m"; make_linear=true)
#opf = load_toy(:rand_qp; n=100, m=30, θ=0.0)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Create and precondition problem
P = EqualityBoxProblem(opf; use_qr=true)

# Algorithm parameters
step = NewtonStep(safety=1.0, solver=:schur_cg, num_cg_iter=100)

# Set up algorithm
history = History(force=[:variable])
@time z, history = barrier_method!(step, P; history=history, ϵ=1e-8, μ=10)

@show length(history.infeasibility)
@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; xscale=:identity)

# using Profile
# Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=15.0, μ=10);