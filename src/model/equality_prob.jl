abstract type EqualityBoxProblem end

struct StandardEqualityBoxProblem <: EqualityBoxProblem
    nlp
end

EqualityBoxProblem(nlp) = StandardEqualityBoxProblem(nlp)

function lagrangian(P::EqualityBoxProblem, x, y)
    return objective(P, x) + y' * constraints(P, x)
end

function primal_residual(P::EqualityBoxProblem, x, y)
    return -constraints(P, x)
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
    jacobian_transpose_product!(rd, P, x, y)
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

function gradient(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    
    ∇f = similar(u)
    grad!(P.nlp, u, ∇f)

    return [∇f; zeros(num_slack_var(P))]
end

function gradient!(∇f, P::EqualityBoxProblem, x)
    n = num_true_var(P)

    # Update main variables
    u = get_true_vars(P, x)
    grad!(P.nlp, u, view(∇f, 1:n))

    # Update slack variables
    ∇f[n+1:end] .= 0

    return ∇f
end

function hessian(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    n, m = num_true_var(P), num_slack_var(P)

    H0 = hess(P.nlp, u)
    return [
        H0 zeros(n, m);
        zeros(m, n) 0*I
    ]
end

function constraints(P::EqualityBoxProblem, x)
    u, s = get_true_vars(P, x), get_slack_vars(P, x)
    m, k = num_slack_var(P), num_eq_con(P)
    
    hu_nlp = zeros(num_con(P))
    cons!(P.nlp, u, hu_nlp)

    hu = similar(hu_nlp)
    hu[1:m] = hu_nlp[_get_ineq_indices(P.nlp)] - s
    hu[m+1:m+k] = hu_nlp[_get_eq_indices(P.nlp)] - get_rhs(P)

    return hu
end

# TODO Optimize
function constraints!(hu, P::EqualityBoxProblem, x)
    hu .= constraints(P, x)
    return hu
end

function jacobian(P::EqualityBoxProblem, x)
    u = get_true_vars(P, x)
    m, k = num_slack_var(P), num_eq_con(P)
    
    J_u = jac(P.nlp, u)

    J_i = J_u[_get_ineq_indices(P.nlp), :]
    J_e = J_u[_get_eq_indices(P.nlp), :]

    return [
        J_i -I(m)
        J_e zeros(k, m)
    ]
end

# TODO Optimize
function jacobian_transpose_product!(Jtv, P::EqualityBoxProblem, x, v)
    J = jacobian(P, x)
    Jtv .= J' * v
    return Jtv
end

function project_box!(P::EqualityBoxProblem, x)
    xmin, xmax = get_box(P)
    @. x = clamp(x, xmin, xmax)
    return x
end

# TODO Optimize
function get_box(P::EqualityBoxProblem)
    j_ineq = _get_ineq_indices(P.nlp)

    x_min = [get_lvar(P.nlp); get_lcon(P.nlp)[j_ineq]]
    x_max = [get_uvar(P.nlp); get_ucon(P.nlp)[j_ineq]]

    return x_min, x_max
end

function get_rhs(P::EqualityBoxProblem)
    return _get_b(P.nlp)
end

# ====
# Indexing
# ====

function get_true_vars(P::EqualityBoxProblem, x)
    return view(x, 1:num_true_var(P))
end

function get_slack_vars(P::EqualityBoxProblem, x)
    return view(x, num_true_var(P)+1:num_true_var(P)+num_slack_var(P))
end

function num_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp) + num_slack_var(P)  # m + n
end

function num_true_var(P::EqualityBoxProblem)
    return get_nvar(P.nlp)  # n
end

function num_slack_var(P::EqualityBoxProblem)
    return length(_get_ineq_indices(P.nlp))  # m
end

function num_con(P::EqualityBoxProblem)
    return get_ncon(P.nlp)  # m + k
end

function num_eq_con(P::EqualityBoxProblem)
    return length(_get_eq_indices(P.nlp))  # k
end

# ====
# NLPModels Helpers
# ====

# TODO Optimize
function _get_eq_indices(nlp)
    return get_jfix(nlp)
end

function _get_ineq_indices(nlp)
    return setdiff(1:get_ncon(nlp), get_jfix(nlp))
end

function _get_b(nlp)
    return get_lcon(nlp)[_get_eq_indices(nlp)]
end







