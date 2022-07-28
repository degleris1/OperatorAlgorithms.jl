include("util.jl")

using Random
Random.seed!(0)

opf = load_dc("case9.m"; make_linear=true)
# opf = load_toy(:bad_constraint)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Parameters
A = HybridGradient
z, ω = 0.9, 10.0
ρ = 0.0
num_iter = 500_000


# Algorithm
P = precondition_cp_ruiz(augment(EqualityBoxProblem(opf), ρ))

η, α = get_good_step(P; z=z, ω=ω)
#η, α = FixedStep(1.0 / ω), FixedStep(1.0 * ω)

alg = A(max_iter=num_iter, η=η, α=α)
history = History(force=[:variable, :y])

@time x, y, history, alg = optimize!(alg, P, history=history)
# @profile optimize!(alg, P, history=history)

@show minimum(history.infeasibility)
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; start=10)

