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
        "case_ACTIVSg2000",
        #"case1354pegase",
    ],
    true => [
    "case30",
    "case118",
    "case300",
    "case_ACTIVSg2000",
    #"case1354pegase",
    ]
)

for use_qr in [false, true]
    for case in CASES[use_qr]
        r = results[(case, use_qr)]
        println("Case: $(case)\t QR: $(use_qr)")
        println("n: $(size(r.prob.A, 2))")
        println("ncg: $(sum(r.hist.cg_iters))")
        println("t: $(r.t)")

        # Get condition number
        H = OperatorAlgorithms.hessian(r.prob, r.z) + safety*I
        if use_qr
            F = qr(sparse(r.prob.A'))
            m, n = size(r.prob.A)
            Q = F.Q  
            Q = F.Q * Matrix(I, n, m)
        else
            Q = r.prob.A'
        end
        K = Q' * H * Q
        cond_num = cond(Matrix(K))
        
        
        println("κ: $(cond_num)")
        println()
    end
end