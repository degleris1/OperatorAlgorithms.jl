struct BarrierProblem <: EqualityBoxProblem
    P::EqualityBoxProblem
    t::Real
end

function Base.getproperty(P::BarrierProblem, s::Symbol)
    if s === :P
        return getfield(P, :P)
    elseif s === :t
        return getfield(P, :t)
    else
        return getfield(P.P, s)
    end
end

function objective(P::BarrierProblem, z::PrimalDual)
    x = z.primal

    f = objective(P.P, z)
    xmin, xmax = get_box(P)

    for i in 1:length(xmin)
        if isfinite(xmin[i])
            f += (x[i] < xmin[i]) ? Inf : -P.t * log(x[i] - xmin[i])
        end
        if isfinite(xmax[i])
            f += (x[i] > xmax[i]) ? Inf : -P.t * log(xmax[i] - x[i])
        end
    end

    return f
end

function gradient(P::BarrierProblem, z::PrimalDual)
    x = z.primal 

    ∇f = gradient(P.P, x)
    xmin, xmax = get_box(P)

    # 1 / Inf = 0
    # So we can just evaluate the vectorized expression
    @. ∇f += -P.t * (1 / (x - xmin))
    @. ∇f += P.t * (1 / (xmax - x))

    return ∇f
end

function gradient!(∇f, P::BarrierProblem, z::PrimalDual)
    x = z.primal

    gradient!(∇f, P.P, z)

    xmin, xmax = get_box(P)

    # 1 / Inf = 0
    # So we can just evaluate the vectorized expression
    @. ∇f += -P.t * (1 / (x - xmin))
    @. ∇f += P.t * (1 / (xmax - x))

    return ∇f
end

function hessian(P::BarrierProblem, z::PrimalDual)
    H = hessian(P.P, x)
    xmin, xmax = get_box(P)

    for i in 1:length(xmin)
        H[i, i] += P.t * (1 / (x[i] - xmin[i]))^2
        H[i, i] += P.t * (1 / (xmax[i] - x[i]))^2
    end

    return H
end

function hessian!(H, P::BarrierProblem, z::PrimalDual)
    error("Not yet supported.")
end
