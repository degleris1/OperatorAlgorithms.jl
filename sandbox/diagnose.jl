include("util.jl")

using Random
Random.seed!(0)

# opf = load_dc("case30.m"; make_linear=true)
opf = load_toy(:rand_qp; n=50, m=10, θ=0.0)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Create and precondition problem
P = EqualityBoxProblem(opf)
P = BarrierProblem(P, 10.0)

# Algorithm parameters
step = GradientStep()
stopping_criteria = FixedStop(max_iter=1000, tol=1e-18)

# Set up algorithm
history = History(force=[:variable])
#z, history = descent!(step, P; history=history, stop=stopping_criteria)

# @show log10(minimum(history.infeasibility))
# @show minimum(distance(history, x_opt)) / norm(x_opt)
# plt = plot_diagnostics(history, x_opt; xscale=:identity)

using Profile
Profile.clear()
@profile descent!(step, P; history=history, stop=stopping_criteria)
