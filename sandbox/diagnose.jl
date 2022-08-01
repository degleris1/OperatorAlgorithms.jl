include("util.jl")

using Random
Random.seed!(0)

opf = load_dc("case30.m"; make_linear=true)
# opf = load_toy(:linear)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = HybridGradient  # algorithm
τ = 1000  # restart every
z = 0.9  # step size
ω = :auto  # primal weight
ρ = 0.0  # augmented term weight
num_iter = 400_000  # number of iterations


# Algorithm
P = precondition_cp_ruiz(augment(EqualityBoxProblem(opf), ρ))

η, α = get_good_step(P; z=z, ω=ω)

subalg = A(max_iter=num_iter, η=η, α=α)
alg = Restarted(opt=subalg, restart_every=τ)
history = History(force=[:variable, :y])

@time x, y, history, alg = optimize!(alg, P, history=history)

@show minimum(history.infeasibility)
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; start=10)

