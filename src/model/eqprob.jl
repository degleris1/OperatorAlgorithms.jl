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

function dual_residual(P::EqualityBoxProblem, x, y)
    ∇f = gradient(P, x)
    Jh = jacobian(P, x)

    return ∇f + Jh' * y
end

function normal_cone(P::EqualityBoxProblem)
    error()
end

function objective(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    return obj(P.nlp, u) 
end

function constraints(P::EqualityBoxProblem, x)
    u, s = get_true_vars(P, x), get_slack_vars(P, x)

    hu = similar(s)
    cons!(P.nlp, u, hu)

    return hu - s
end

function gradient(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    
    ∇f = similar(u)
    grad!(P.nlp, u, ∇f)

    return [∇f; zeros(num_con(P))]
end

function jacobian(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    
    J = jac(P.nlp, u)
    return hcat(J, -I)
end

function project_box!(P::EqualityBoxProblem, x)
    x_min, x_max = get_box(P)
    x .= clamp.(x, x_min, x_max)

    return x
end

function initialize(P::EqualityBoxProblem)
    return project_box!(P, [get_x0(P.nlp); zeros(num_con(P))])
end

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

function num_con(P::EqualityBoxProblem)
    return get_ncon(P.nlp)
end
