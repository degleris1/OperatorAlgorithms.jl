# ====
# Optimizer History
# ====

Base.@kwdef mutable struct History
    primal_residual = []
    dual_residual = []
    primal_infeasibility::Vector{Real} = []
    dual_infeasibility::Vector{Real} = []
    objective::Vector{Real} = []
    variable = []
    force::Vector{Symbol} = []
    num_iter::Int = 0
end

function Base.getproperty(h::History, s::Symbol)
    if s == :infeasibility
        return sqrt.(h.primal_infeasibility .^ 2 + h.dual_infeasibility .^ 2)
    else
        return getfield(h, s)
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

function update!(history::History, prob, z, alg_info)

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

    for (k, f) in zip(KEYS, fs)
        if k in keys(alg_info)
            data = getproperty(history, k)
            push!(data, getproperty(alg_info, k))
        elseif k in history.force
            data = getproperty(history, k)
            push!(data, f(prob, z))
        end
    end

    history.num_iter += 1

    return history
end


