include("util.jl")

using Random
Random.seed!(0)

# opf = load_dc("case30.m"; make_linear=true)
opf = load_toy(:rand_qp; n=300, m=30, θ=0.0)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = (; kwargs...) -> TruncNewton(; k=10, kwargs...)
ω = 1.0  # primal weight, 1.0 is default (automatic)
num_iter = 100  # number of iterations

# Create and precondition problem
P = EqualityBoxProblem(opf, ω)
P = BarrierProblem(P, 1.0)

# Set up step sizes
η, α = FixedStep(1.0), FixedStep(1.0)

# Set up algorithm
alg = A(max_iter=num_iter, η=η, α=α)
history = History(force=[:variable])
@time x, y, history, alg = optimize!(alg, P, history=history)

@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; xscale=:identity)


