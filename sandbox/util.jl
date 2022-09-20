using Pkg; Pkg.activate(@__DIR__)

using LinearAlgebra
using SparseArrays
using MadNLP
using Random
using Plots

import NLPModels

using Revise
using OperatorAlgorithms

fnt = (Plots.GR.FONT_TIMES_ROMAN, 8)
gr()
theme(:default; 
    label=nothing, 
    titlefont=fnt, 
    guidefont=fnt, 
    tickfont=fnt, 
    legendfontsize=8,
)
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

function plot_diagnostics(history, x_opt; height=5, width=20, start=1, xscale=:identity)
    EPS = 1e-15
    ucp = (
        xscale=xscale,
        lw=2,
        linetype=:steppost
    )

    x = cumsum(history.cg_iters)
    pinf = history.primal_infeasibility
    dinf = history.dual_infeasibility
    tinf = history.infeasibility
    
    plt1 = plot(x, pinf .+ EPS; ylabel="Primal Infeasibility", yscale=:log10, ucp...)
    plot!(plt1, [x[1]; x[history.num_step]], [pinf[1]; history.true_pinf]; ucp...)

    plt2 = plot(x, dinf .+ EPS; ylabel="Dual Infeasibility", yscale=:log10, ucp...)
    plot!(plt2, [x[1]; x[history.num_step]], [dinf[1]; history.true_dinf]; ucp...)
    plot!(plt2, xlabel="Num CG Iters")

    # plt4 = plot(x, tinf .+ EPS; ylabel="Total Infeasibility", yscale=:log10, ucp...)
    # if :variable in keys(history.data)
    #     plt3 = plot(x, distance(history, x_opt) / length(x_opt);  ylabel="rms(x - x*)", ucp...)
    # else
    #     plt3 = deepcopy(plt4)
    # end

    plt = plot(plt1, plt2, layout=(2, 1))

    return plt
end


