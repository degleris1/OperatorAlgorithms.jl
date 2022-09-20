include("util.jl")
using JLD
using StatsPlots
using LaTeXStrings
gr()

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 500_000
safety = 1.0
block_size = 256
obj_scale = 1.0

results = JLD.load("/Users/degleris/Data/opf_results.jld")["results"]

# Solve with QR
CASES = [
    "case30",
    "case118",
    "case300",
]

plts = []
for case in CASES
    r = results[(case, true)]
    m, n = size(r.prob.A)

    H = OperatorAlgorithms.hessian(r.prob, r.z) + safety*I
    A = r.prob.A
    Q = qr(sparse(A')).Q * Matrix(I, n, m)

    AHAt = A * inv(H) * A'
    QtHQ = Q' * inv(H) * Q

    s1 = svdvals(inv(H))
    s2 = svdvals(Matrix(AHAt))
    s3 = svdvals(Matrix(QtHQ))

    legend = (case == "case300") ? true : false
    ylabel = (case == "case30") ? "Singular Value" : ""
    yticks = (case == "case30") ? 10.0 .^ (-4:2:4) : []

    plt = plot(
        title=case, 
        legend=legend, 
        ylabel=ylabel, 
        yticks=yticks,
        ylim=(10^(-5), 10^5),
        yscale=:log10,
        grid=false,
    )

    kwargs = (lw=3, la=0.75)
    plot!(plt, s1; label=L"H^{-1}", kwargs...)
    plot!(plt, s2; label=L"A H^{-1} A^T", kwargs...)
    plot!(plt, s3; label=L"Q^T H^{-1} Q", kwargs...)

    push!(plts, plt)
end

fig = plot(plts..., layout=(1, 3), size=(650, 200))
savefig("/Users/degleris/Data/spectra.pdf")
fig