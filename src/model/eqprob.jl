abstract type EqualityBoxProblem end

struct StandardEqualityBoxProblem <: EqualityBoxProblem
    nlp
end

EqualityBoxProblem(nlp) = StandardEqualityBoxProblem(nlp)

function lagrangian(P::EqualityBoxProblem, x, y)
    return objective(P, x) + y' * constraints(P, x)
end

function primal_residual(P::EqualityBoxProblem, x, y)
    return constraints(P, x)
end

function primal_residual!(rp, P::EqualityBoxProblem, x, y)
    constraints!(rp, P, x)
    @. rp = -rp
    return rp
end

function dual_residual(P::EqualityBoxProblem, x, y)
    ∇f = gradient(P, x)
    Jh = jacobian(P, x)

    return ∇f + Jh' * y
end

function dual_residual!(rd, P::EqualityBoxProblem, x, y, g)
    # Compute ∇f + Jh' y
    
    # First, compute Jh' * y
    jacobian_transpose_product!(rd, P, x, y)

    # Then add ∇f
    rd .+= g

    return rd
end

# TODO Optimize
function normal_cone_distance(P::EqualityBoxProblem, x, v; tol=1e-5)
    x_min, x_max = get_box(P)
 
    # Find binding constraints
    not_binding = @. (abs(x_min - x) > tol) & (abs(x_max - x) > tol)

    # Compute error in the non-bound components
    return norm(v[not_binding])
end

function objective(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    return obj(P.nlp, u) 
end

# TODO
# rethink constraints
# need to either add a barrier to slack vars
# or to solve equality vars directly

# basically, we have the following problem
# min_{x, s} f(x)
# Ax - b = 0   (true eq vars)
# Fx - s = 0   (ineq vars)
# x in X  (box constraints on x)
# s in S  (box constraints on s)

# box constraints are a tad annoying...
# maybe i should convert it to a standard form convex problem first
# then introduce the slack vars (if desired)

function constraints(P::EqualityBoxProblem, x)
    u, s = get_true_vars(P, x), get_slack_vars(P, x)

    hu = similar(s)
    cons!(P.nlp, u, hu)

    j_eq = get_jfix(P.nlp)
    j_var = setdiff(1:length(u), j_eq)

    b = get_lcon(P.nlp)[j_eq]

    return [
        hu[j_eq] - b;
        hu[j_var] - s
    ]
end

function constraints!(hu, P::EqualityBoxProblem, x)
    #u, s = get_true_vars(P, x), get_slack_vars(P, x)
    #cons!(P.nlp, u, hu)
    #hu .-= s
    
    hu .= constraints(P, x)

    return hu
end

function gradient(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    
    ∇f = similar(u)
    grad!(P.nlp, u, ∇f)

    return [∇f; zeros(num_con(P))]
end

function gradient!(∇f, P::EqualityBoxProblem, x)
    n = num_true_var(P)

    # Update main variables
    u = view(x, 1:n)
    grad!(P.nlp, u, view(∇f, 1:n))

    # Update slack variables
    ∇f[n+1:end] .= 0

    return x
end

function hessian(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    n, m = length(u), num_con(P)

    H0 = hess(P.nlp, u)
    return [
        H0 zeros(n, m);
        zeros(m, n) 0*I
    ]
end

function jacobian(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    
    J = jac(P.nlp, u)
    return hcat(J, -I)
end

function jacobian_transpose_product!(Jtv, P::EqualityBoxProblem, x, v)
    u, s = get_true_vars(P, x), get_slack_vars(P, x)
    n = num_true_var(P)

    jtprod!(P.nlp, u, v, view(Jtv, 1:n))
    Jtv[n+1:end] .= s

    return Jtv
end

function project_box!(P::EqualityBoxProblem, x)
    n = num_true_var(P)
    m = num_con(P)

    x[1:n] .= clamp.(view(x, 1:n), get_lvar(P.nlp), get_uvar(P.nlp))
    x[n+1:end] .= clamp.(view(x, n+1:n+m), get_lcon(P.nlp), get_ucon(P.nlp))

    return x
end

#function initialize(P::EqualityBoxProblem)
#    return project_box!(P, [get_x0(P.nlp); zeros(num_con(P))])
#end

function get_box(P::EqualityBoxProblem)
    x_min = [get_lvar(P.nlp); get_lcon(P.nlp)]
    x_max = [get_uvar(P.nlp); get_ucon(P.nlp)]
    return x_min, x_max
end

function get_true_vars(P::EqualityBoxProblem, x)
    return view(x, 1:num_true_var(P))
end

function get_slack_vars(P::EqualityBoxProblem, x)
    return view(x, num_true_var(P)+1:num_true_var(P)+num_con(P))
end

function num_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp) + get_ncon(P.nlp)
end

function num_true_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp)
end

function num_slack_var(P::EqualityBoxProblem)
    
end

function num_con(P::EqualityBoxProblem)
    return get_ncon(P.nlp)
end
