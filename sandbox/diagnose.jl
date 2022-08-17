include("util.jl")

using Random
Random.seed!(0)

opf = load_dc("case30.m"; make_linear=true)
#opf = load_toy(:rand_qp; n=100, m=30, θ=0.0)

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Create and precondition problem
P = EqualityBoxProblem(opf)
P = BarrierProblem(P, 0.001)

# Algorithm parameters
#step = NewtonStep(safety=0.01, solver=:schur_cg, use_qr=true, num_cg_iter=100)
step = NewtonStep(safety=1e-1)
stopping_criteria = FixedStop(max_iter=200, tol=1e-10)

# Set up algorithm
history = History(force=[:variable])
@time z, history = descent!(step, P; history=history, stop=stopping_criteria)

@show log10(minimum(history.infeasibility))
@show minimum(distance(history, x_opt)) / norm(x_opt)
plt = plot_diagnostics(history, x_opt; xscale=:identity)