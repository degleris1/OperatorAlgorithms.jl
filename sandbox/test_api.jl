using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra

using Revise
using OperatorAlgorithms

nlp = load_case("case9.m")

alg = Dommel(max_iter=10_000, η=1e-4, ρ=1e4)
u, λ, history, alg = optimize!(alg, nlp)

@show norm.(history.dual_residual)[end-4:end]
@show norm.(history.primal_residual)[end-4:end]
;
