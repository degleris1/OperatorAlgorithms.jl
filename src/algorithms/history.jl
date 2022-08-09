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
    x = []
    y = []
    force::Vector{Symbol} = []
    num_iter::Int = -1
end

function Base.getproperty(h::History, s::Symbol)
    if s == :infeasibility
        return sqrt.(h.primal_infeasibility .^ 2 + h.dual_infeasibility .^ 2)
    else
        return getfield(h, s)
    end
end

function initialize!(history::History, prob, x, y)
    update!(history, prob, x, y, (;))
end

function distance(history::History, x_opt)
    return [norm(xi[1:length(x_opt)] - x_opt) for xi in history.variable]
end

function primal_infeasibility(p, x, y)
    return norm(primal_residual(p, x, y)) / norm(x)
end

function dual_infeasibility(p, x, y)
    return normal_cone_distance(p, x, dual_residual(p, x, y)) / norm(y)
end

function update!(history::History, prob, x, y, alg_info)

    KEYS = [
        :primal_residual, :dual_residual, 
        :objective, :variable, :x, :y,
        :primal_infeasibility, :dual_infeasibility,
    ]
    fs = [
        primal_residual, 
        dual_residual, 
        (p, x, y) -> objective(p, x), 
        (p, x, y) -> Array(get_true_vars(p, x)),
        (p, x, y) -> deepcopy(x),
        (p, x, y) ->  deepcopy(y),
        primal_infeasibility,
        dual_infeasibility,
    ]

    for (k, f) in zip(KEYS, fs)
        if k in keys(alg_info)
            data = getproperty(history, k)
            push!(data, getproperty(alg_info, k))
        elseif k in history.force
            data = getproperty(history, k)
            push!(data, f(prob, x, y))
        end
    end

    history.num_iter += 1

    return history
end


