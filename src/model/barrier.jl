struct BarrierProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    t::Real
end

function Base.getproperty(P::BarrierProblem, s::Symbol)
    if s === :nlp
        return P.P.nlp
    else
        return getfield(P, s)
    end
end

function objective(P::BarrierProblem, x)
    f = P.t * objective(P.P, x)
    xmin, xmax = get_box(P.P)

    for i in 1:length(xmin)
        if isfinite(xmin[i])
            f += (x[i] < xmin[i]) ? Inf : -log(x[i] - xmin[i])
        end
        if isfinite(xmax[i])
            f += (x[i] > xmax[i]) ? Inf : -log(xmax[i] - x[i])
        end
    end

    return f
end

function gradient(P::BarrierProblem, x)
    ∇f = P.t * gradient(P.P, x)
    xmin, xmax = get_box(P.P)

    # 1 / Inf = 0
    # So we can just evaluate the vectorized expression
    @. ∇f += -(1 / (x - xmin))
    @. ∇f += (1 / (xmax - x))

    return ∇f
end

# TODO Optimize
function gradient!(∇f, P::BarrierProblem, x)
    ∇f .= gradient(P, x)
    return ∇f
end

function hessian(P::BarrierProblem, x)
    H = P.t * hessian(P.P, x)
    xmin, xmax = get_box(P.P)

    for i in 1:length(xmin)
        H[i, i] += (1 / (x[i] - xmin[i]))^2
        H[i, i] += (1 / (xmax[i] - x[i]))^2
    end

    return H
end

