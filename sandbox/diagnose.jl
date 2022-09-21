include("util.jl")

Random.seed!(0)

# Parameters
rtol = 1e-4
μ = 5
max_cg_iter = 25_000
safety = 1.0
block_size = 1024
obj_scale = 1.0
precond = :diag

# case1354pegase, case2869pegase, case13659pegase
# case_ACTIVSg2000, case_ACTIVSg10k, case_ACTIVSg70k
opf = load_dc("matpower/case118.m")

# Baseline
stats = madnlp(opf; print_level=MadNLP.WARN)
x_opt = stats.solution
scale = norm(NLPModels.grad(opf, x_opt))
println("MadNLP Time: $(stats.elapsed_time)")
println("Newton Steps: $(stats.iter)\n")

# Create problem
P = BoxQP(opf; use_qr=true, block_size=block_size, scale=obj_scale)
stop = FixedStop(100, rtol*scale)

# Primal dual method
step = NewtonStep(primal_dual=true, safety=1.0, solver=:schur_cg, num_cg_iter=max_cg_iter, precond=precond)
runtime = @elapsed z, history = descent!(step, P; stop=stop, init_dual=true, verbose=1)
println("QRCG Time: $(runtime)")
println("Newton Steps: $(history.num_iter), CG Steps: $(sum(history.cg_iters))\n")

pstar = NLPModels.obj(opf, z.primal[1:length(x_opt)])
@show log10(minimum(history.infeasibility) / scale)
@show log10( abs(pstar - stats.objective) / stats.objective )

# Barrier method
# step = NewtonStep(safety=10.0, solver=:schur_cg, num_cg_iter=max_cg_iter)
# @time z, history = barrier_method!(step, P; ϵ=rtol*scale, μ=μ, verbose=1)
# @show history.num_iter, sum(history.cg_iters)

#plt = plot_diagnostics(history, x_opt)

# using Profile; Profile.clear()
# @profile barrier_method!(step, P; history=history, ϵ=1e-2*scale, μ=5)
