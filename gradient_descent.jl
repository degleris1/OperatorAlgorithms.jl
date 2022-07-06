using Artifacts
using Argos
using Ipopt
using LinearAlgebra

MOI = Argos.MOI

datafile = joinpath(artifact"ExaData", "ExaData", "case9.m")
nlp_opt = Argos.ReducedSpaceEvaluator(datafile)


# Baseline solution
opt = Ipopt.Optimizer()
MOI.set(opt, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")
MOI.set(opt, MOI.RawOptimizerAttribute("tol"), 1e-4)
MOI.set(opt, MOI.RawOptimizerAttribute("print_level"), 0)

solution = Argos.optimize!(opt, nlp_opt)
u_opt = solution.minimizer
p_opt = solution.minimum

@show p_opt
println()

# Initialize
nlp = Argos.ReducedSpaceEvaluator(datafile)
u = Argos.initial(nlp)
Argos.update!(nlp, u)

@show Argos.objective(nlp, u)
println()

function feasibility(nlp, u)
    return [
        nlp.g_min - Argos.constraint(nlp, u)
        Argos.constraint(nlp, u) - nlp.g_max
    ]
end

# Descent
function descent!(u, nlp; η=1e-5, α=1e-6, ρ=50.0, num_iter=550)
    J0 = Argos.jacobian(nlp, u)
    m, n = size(J0)
    λmin, λmax = zeros(m), zeros(m)

    history = Real[]
    for iter in 1:num_iter
        # Constraint penalty
        J = Argos.jacobian(nlp, u)
        h = max.(0, feasibility(nlp, u))
        penalty = ρ * J' * (h[1:m] + h[m+1:end]) + J' * (λmax - λmin)
        
        # ∇f
        ∇f = Argos.gradient(nlp, u)
        
        # Descend and project
        u .-= η * (∇f + penalty)
        u .= clamp(u, nlp.u_min, nlp.u_max)

        Argos.update!(nlp, u)
        push!(history, Argos.objective(nlp, u))
        
        # Update dual variables
        hu = Argos.constraint(nlp, u)
        λmin .+= α * (nlp.g_min - hu)
        λmin .= max.(0, λmin)
        λmax .+= α * (hu - nlp.g_max)
        λmax .= max.(0, λmax)
    end
    return u, history
end

u, history = descent!(u, nlp)
@show Argos.objective(nlp, u)
@show maximum(max.(0, feasibility(nlp, u)))
@show norm(u - u_opt) / length(u)
