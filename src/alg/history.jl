# ====
# Optimizer History
# ====

Base.@kwdef mutable struct History
    data::Dict{Symbol, Any} = Dict{Symbol, Any}()
    force::Vector{Symbol} = []
    num_iter::Int = 0
end

function Base.getproperty(h::History, s::Symbol)
    if hasfield(typeof(h), s)
        return getfield(h, s)
    else
        return get(() -> error("No data on $(s)"), h.data, s)
    end
end

function distance(history::History, x_opt)
    return [norm(zi.primal[1:length(x_opt)] - x_opt) for zi in history.variable]
end

function primal_infeasibility(p, z)
    return norm(primal_residual(p, z))
end

function dual_infeasibility(p, z)
    return norm(dual_residual(p, z))
end

function update!(history::History, prob, z, alg_info; should_inc=true, should_force=true)

    KEYS = [
        :primal_residual, :dual_residual, 
        :objective, :variable,
        :primal_infeasibility, :dual_infeasibility,
    ]
    fs = [
        primal_residual, 
        dual_residual, 
        objective, 
        (p, z) -> deepcopy(z),
        primal_infeasibility,
        dual_infeasibility,
    ]

    # Add algorithm info
    for k in keys(alg_info)
        hist = get!(history.data, k, [])
        push!(hist, getproperty(alg_info, k))
    end

    # Add extra info
    if should_force
        for (k, f) in zip(KEYS, fs)
            if (k in history.force) && !(k in keys(alg_info))
                hist = get!(history.data, k, [])
                push!(hist, f(prob, z))
            end
        end
    end

    if should_inc
        history.num_iter += 1
    end

    return history
end


