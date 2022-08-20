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

function hessian!(H::Diagonal, P::BarrierProblem, z::PrimalDual)
    hessian!(H, P.P, z)

    x = z.primal
    xmin, xmax = get_box(P)

    @. H.diag += P.t * (1 / (x - xmin))^2
    @. H.diag += P.t * (1 / (xmax - x))^2

    return H
end

function get_multipliers(P::BarrierProblem, z::PrimalDual)
    # We have t = fi * λi
    # So λi = t / fi

    x = z.primal
    xmin, xmax = get_box(P)

    λmin = P.t ./ (x - xmin)
    λmax = P.t ./ (xmax - x)

    λmin[λmin .== Inf] .= 0
    λmax[λmax .== Inf] .= 0

    return λmin, λmax
end