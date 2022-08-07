using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using Plots
gr()
theme(:default; label=nothing, titlefontsize=10, guidefontsize=10, tickfontsize=8)

using Revise
using OperatorAlgorithms

OperatorAlgorithms.PowerModels.Memento.config!("error")

function rolling(x, f=minimum)
    return [f(x[1:k]) for k in 1:length(x)]
end

function print_diagnostics(h, x_opt)
    @show h.dual_infeasibility[[1, end]]
    @show h.primal_infeasibility[[1, end]]
    @show distance(h, x_opt)[[1, end]]
end

function crd(i, data)
    return [x[i] for x in data]
end

function plot_diagnostics(history, x_opt; height=5, width=20, start=1, xscale=:log10)
    ucp = (
        xticks=10 .^ (0:floor(Int, log10.(length(history.primal_infeasibility)))),
#        xticks=[0, 0.9*length(history.primal_infeasibility)],
        xlim=(1, length(history.primal_infeasibility)),
        xscale=xscale,
    )
    plt = plot(
        plot(history.primal_infeasibility[start:end]; 
             ylabel="Primal Infeasibility", yscale=:log10, ucp...),
        plot(history.dual_infeasibility[start:end] .+ 1e-15; 
             ylabel="Dual Infeasibility", yscale=:log10, ucp...),
        plot(distance(history, x_opt)[start:end] / length(x_opt); 
             ylabel="rms(x - x*)", ucp...),
        plot(history.infeasibility[start:end]; 
             ylabel="Total Infeasibility", yscale=:log10, ucp...),
        layout = (2, 2),
    )
    return plt
end


