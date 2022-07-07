# Useful functions that build upon the NLPModels API

function get_objective(nlp, u, λ)
    return obj(nlp, u)
end

function get_dual_residual(nlp, u, λ)
    J = inequality_jacobian(nlp, u)
    
    ∇f = similar(u)
    grad!(nlp, u, ∇f)
    
    return ∇f + J' * λ
end

function get_primal_residual(nlp, u, λ)
    return max.(0, inequality_constraints(nlp, u))
end

"""
    project!(nlp, u, λ)

Project `u` onto the box `[u_min, u_max]` and `λ` onto the nonnegative orthant.
"""
function project!(nlp, u, λ)
    u_min, u_max = get_lvar(nlp), get_uvar(nlp)
    u .= clamp.(u, u_min, u_max)

    λ .= max.(0, λ)

    return u, λ
end

"""
    inequality_constraints(nlp, u)

Return the evaluated constraints in standard form, `h(u) <= 0`.
"""
function inequality_constraints(nlp, u)
    c_min, c_max = get_lcon(nlp), get_ucon(nlp)

    # Get indices with real constraints (not Inf)
    jl, ju = _inequality_indices(nlp)  

    # Throw an error if there are equality constraints
    # (Currently unsupported)
    @assert length(nlp.meta.jfix) == 0

    hu = zeros(nlp.meta.ncon)
    cons!(nlp, u, hu)

    return [
        c_min[jl] - hu[jl];
        hu[ju] - c_max[ju];
    ]
end

"""
    inequality_jacobian(nlp, u)
"""
function inequality_jacobian(nlp, u)
    jl, ju = _inequality_indices(nlp)
    J = jac(nlp, u)

    return [
        -J[jl, :];
        J[ju, :];
    ]
end

function gradient(nlp, u)
    ∇f = similar(u)
    grad!(nlp, u, ∇f)
    return ∇f
end

function dim_constraints(nlp)
    jl, ju = _inequality_indices(nlp)
    return length(jl) + length(ju)
end

function dim_variables(nlp)
    return get_nvar(nlp)
end

function get_var(nlp, u, λ)
    return deepcopy(u), deepcopy(λ)
end

function _inequality_indices(nlp)
    jl, jm, ju = nlp.meta.jlow, nlp.meta.jrng, nlp.meta.jupp

    return [jl; jm], [jm; ju]
end





