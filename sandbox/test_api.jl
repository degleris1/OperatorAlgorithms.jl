using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using UnicodePlots

using Revise
using OperatorAlgorithms

opf = load_dc("case3.m")
P = EqualityBoxProblem(opf)

alg = Dommel(max_iter=100_000, η=1e-5, ρ=1e4, max_rel_step_length=1.0)
x, y, history, alg = optimize!(alg, P)

@show last(norm.(history.alg))
@show norm.(history.dual_residual)[[1, 2, end-1, end]]
@show norm.(history.primal_residual)[[1, 2, end-1, end]]
lineplot(norm.(history.alg))
