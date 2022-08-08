include("util.jl")

using Random
Random.seed!(0)

#opf = load_dc("case30.m"; make_linear=true)
opf = load_toy(:barrier; n=100, m=10, θ=0.0, μ=1.0)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = Newton
τ = 1e10  # restart every
z = 1.0  # step size
ω = 1.0  # primal weight, 1.0 is default (automatic)
ρ = 10.0  # augmented term weight
num_iter = 50  # number of iterations
trust = 1.0

# Create and precondition problem
P = EqualityBoxProblem(opf)

# Set up step sizes
η, α = get_good_step(P; z=z, ω=ω)
η = TrustStep(η, trust/η.η)
α = TrustStep(α, trust/α.η)

η, α = FixedStep(z), FixedStep(z)

# Set up algorithm
alg = A(max_iter=num_iter, η=η, α=α)
cont = Continuation(opt=alg, _weight=1.0)
rest = Restarted(opt=alg, restart_every=τ)
history = History(force=[:variable])

@time x, y, history, alg = optimize!(alg, P, history=history)

@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; xscale=:identity)


