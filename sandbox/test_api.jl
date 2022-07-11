using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using UnicodePlots

using Revise
using OperatorAlgorithms

opf = load_dc("case3.m")

# Baseline
# stats = OperatorAlgorithms.ipopt(opf)

# Algorithm
P = augment(EqualityBoxProblem(opf), 1e4)
alg = Dommel(max_iter=10_000, η=2e-5, max_rel_step_length=1.0, update_dual=true)
x, y, history, alg = optimize!(alg, P)

@show last(norm.(history.alg))
@show norm.(history.dual_residual)[[1, 2, end-1, end]]
@show norm.(history.primal_residual)[[1, 2, end-1, end]]
lineplot(norm.(history.alg))
