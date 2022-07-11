using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using Plots
unicodeplots()

using Revise
using OperatorAlgorithms

function rolling(x, f=minimum)
    return [f(x[1:k]) for k in 1:length(x)]
end

function print_diagnostics(h, x_opt)
    @show h.dual_infeasibility[[1, end]]
    @show h.primal_infeasibility[[1, end]]
    @show distance(h, x_opt)[[1, end]]
end

function plot_diagnostics(history, x_opt, height=5, width=30, start=10)
    ucp = (extra_kwargs=Dict(:subplot => (; height=height, width=width)), ylim=(-0.1, Inf))
    plt = plot(
        plot(history.primal_infeasibility[start:end]; ylabel="pinf", ucp...),
        plot(history.dual_infeasibility[start:end]; ylabel="dinf", ucp...),
        plot(distance(history, x_opt)[start:end]; ylabel="x - x_opt", ucp...),
        layout = (3, 1),
    )
    return plt
end


