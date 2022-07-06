# Useful functions that build upon the NLPModels API

"""
    project_onto_box!(nlp, u)

Project `u` onto the box `[u_min, u_max]`.
"""
function project_onto_box!(nlp, u)
    u_min, u_max = get_lvar(nlp), get_uvar(nlp)
    u .= clamp.(u, u_min, u_max)
    return u
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
    @assert length(nlp.jfix) == 0

    hu = cons(nlp, u)
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

function dim_constraints(nlp)
    jl, ju = _inequality_indices(nlp)
    return length(jl) + length(ju)
end

function dim_variables(nlp)
    return get_nvar(nlp)
end

function _inequality_indices(nlp)
    jl, jm, ju = nlp.meta.jlow, nlp.meta.jrng, nlp.meta.jupp

    return [jl; jm], [jm; ju]
end





