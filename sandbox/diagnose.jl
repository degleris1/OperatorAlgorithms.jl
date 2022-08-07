include("util.jl")

using Random
Random.seed!(0)

#opf = load_dc("case30.m"; make_linear=true)
opf = load_toy(:barrier; n=200, m=10, θ=0.0, μ=0.1)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = HybridGradient
τ = 1e10  # restart every
z = 1e-2  # step size
ω = 1.0  # primal weight, 1.0 is default (automatic)
ρ = 10.0  # augmented term weight
num_iter = 1000  # number of iterations
trust = 1.0

# Create and precondition problem
P = precondition_cp_ruiz(EqualityBoxProblem(opf))
P = OperatorAlgorithms.RegularizedProblem(P, 1e-8)

# Set up step sizes
η, α = get_good_step(P; z=z, ω=ω)
η = TrustStep(η, trust/η.η)
α = TrustStep(α, trust/α.η)

# Set up algorithm
alg = A(max_iter=num_iter, η=η, α=α)
cont = Continuation(opt=alg, _weight=1.0)
rest = Restarted(opt=alg, restart_every=τ)
history = History(force=[:variable])

@time x, y, history, alg = optimize!(alg, P, history=history)

@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt)


