include("util.jl")

opf = load_dc("case300.m")

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = Dommel
ρ = 1e0
η = 1e-7 / (1+sqrt(ρ))
α = 1.0

# Algorithm
history = History(ignore=[:x, :y])
P = augment(EqualityBoxProblem(opf), ρ)
alg = A(max_iter=10_000, η=η, α=α, max_rel_step_length=1.0)
@time x, y, history, alg = optimize!(alg, P, history=history)

@show minimum(history.dual_infeasibility[end-100:end])
@show minimum(history.primal_infeasibility[end-100:end])
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt)

