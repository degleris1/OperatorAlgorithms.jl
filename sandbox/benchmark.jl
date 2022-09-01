include("util.jl")

using Random
Random.seed!(0)

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 500
safety = 1.0
block_size = 1024

CASES = [
    "case300",
    "case1354pegase",
    "case2869pegase",
    "case_ACTIVSg2000",
]

function solve_case(case; verbose=true)
    opf = load_dc("matpower/$(case).m")
    P = BoxQP(opf; use_qr=true, block_size=block_size)
    scale = norm(OperatorAlgorithms.gradient(P, OperatorAlgorithms.initialize(P)))
    
    history = History()
    step = NewtonStep(safety=safety, solver=:schur_cg, num_cg_iter=max_cg_iter)
    τ = @elapsed z, history = barrier_method!(step, P; history=history, ϵ=rtol*scale, μ=μ)

    r = (
        scale=scale,
        solve_time=τ, 
        dinf=last(history.true_dinf), 
        pinf=last(history.true_pinf), 
        num_newton=history.num_iter, 
        num_cg=sum(history.cg_iters)
    )

    @show case, τ, r.num_newton, r.num_cg

    return r
end

# Compile
solve_case("case9")

# Benchmark
results = solve_case.(CASES)