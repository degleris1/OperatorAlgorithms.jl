include("util.jl")

Random.seed!(0)

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 500_000
safety = 1.0
block_size = 256
obj_scale = 1.0

# Solve with QR
CASES = Dict(
    false => [
        "case30",
        "case118",
        "case300",
        "case1354pegase",
        "case_ACTIVSg2000",
    ],
    true => [
    "case30",
    "case118",
    "case300",
    "case1354pegase",
    "case2869pegase",
    "case13659pegase",
    "case_ACTIVSg2000",
    ]
)

results = Dict()
for use_qr in [false, true]
    for case in CASES[use_qr]
        println("Case: $case. QR: $(use_qr)")

        opf = load_dc("matpower/$(case).m")
        stats = madnlp(opf; print_level=MadNLP.WARN)
        scale = norm(NLPModels.grad(opf, stats.solution))

        P = BoxQP(opf; use_qr=use_qr, block_size=block_size, scale=obj_scale)
        step = NewtonStep(primal_dual=true, safety=safety, solver=:schur_cg, num_cg_iter=max_cg_iter)
        runtime = @elapsed z, history = descent!(step, P; 
            stop=FixedStop(100, rtol*scale), init_dual=true, verbose=1)

        results[(case, use_qr)] = (prob=P, z=z, hist=history, t=runtime)
    end
end