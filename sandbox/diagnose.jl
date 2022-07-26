include("util.jl")

using Random
Random.seed!(0)

# opf = load_dc("case118.m")
opf = load_toy(:linear)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = HybridGradient
ρ = 0.0
η = 1e-1
α = 1.0
num_iter = 1000

# Other parameters


# Algorithm
history = History(force=[:variable, :x, :y, :dual_residual])
P = augment(EqualityBoxProblem(opf), ρ)
alg = A(max_iter=num_iter, η=η, α=α, max_rel_step_length=Inf)
@time x, y, history, alg = optimize!(alg, P, history=history)
# @profile optimize!(alg, P, history=history)

@show minimum(history.infeasibility)
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; start=10)

