include("util.jl")

opf = load_dc("case3.m")

# Baseline
stats = solve_ipopt(opf)
x_opt = stats.solution

# Algorithm
P = EqualityBoxProblem(opf)  # augment(EqualityBoxProblem(opf), 1)
alg = ExtraGrad(max_iter=10_000, η=1e-3, max_rel_step_length=Inf)
x, y, history, alg = optimize!(alg, P)

print_diagnostics(history, x_opt)
plt = plot_diagnostics(history, x_opt)

