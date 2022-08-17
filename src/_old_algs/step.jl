abstract type AbstractStep end

function step!(s::AbstractStep, x, dx)
    @. x -= s.η * dx
    return x
end

"""
    FixedStep <: AbstractStep

Fixed primal and dual step sizes `α` and `η`.
"""
struct FixedStep <: AbstractStep
    η::Real
end

"""
    AdaptiveStep <: AbstractStep
"""
mutable struct AdaptiveStep <: AbstractStep
    βp::Real
    βm::Real
    η::Real
    _norm_dx::Real
end

AdaptiveStep(η; βp=1.05, βm=0.8) = AdaptiveStep(βp, βm, η, Inf)

function step!(s::AdaptiveStep, x, dx)
    # Adjust step size
    ndx = norm(dx)
    if ndx <= (1+1e-4) * s._norm_dx
        s.η *= s.βp
    else
        s.η *= s.βm
    end
    s._norm_dx = ndx

    # Update
    @. x -= s.η * dx
    return x
end

function get_good_step(P::EqualityBoxProblem; ω=1, z=0.9, ϵ1=0, ϵ2=1.0)
    x, y = initialize(P)
    xmin, xmax = get_box(P)

    η = z / (opnorm(Array(jacobian(P, x))) + 1e-4)
    
    ndx = norm(gradient(P, x))
    ndy = norm(xmin[xmin .!= -Inf]) + norm(xmax[xmax .!= Inf])

    if ndy > ϵ1
        ω0 = (ndx + ϵ2) / (ndy + ϵ2) 
    else
        ω0 = 1
    end
    ω *= ω0

    return FixedStep(η / ω), FixedStep(η * ω)
end


"""
    TrustStep <: AbstractStep
"""
mutable struct TrustStep <: AbstractStep
    step::AbstractStep
    trust::Real
end

function step!(s::TrustStep, x, dx)
    
    # Shrink step if needed
    ndx = norm(dx)
    if ndx >= s.trust
        @. dx *= s.trust / ndx
    end

    # Update
    return step!(s.step, x, dx)
end

mutable struct RMSProp <: AbstractStep
    η
    ρ
    ϵ
    acc
end

RMSProp(; η = 0.001, ρ = 0.99, ϵ = 1e-8) = RMSProp(η, ρ, ϵ, IdDict())

function step!(s::RMSProp, x, dx)
    (; η, ρ) = s
    acc = get!(() -> zero(x), s.acc, x)
    @. acc = ρ * acc + (1 - ρ) * dx * dx
    @. x -= η * dx / (sqrt(acc) + s.ϵ)
end

mutable struct Momentum <: AbstractStep
    η
    ρ
    velocity
end

Momentum(; η = 0.001, ρ = 0.9) = Momentum(η, ρ, IdDict())

function step!(s::Momentum, x, dx)
    (; η, ρ) = s
    v = get!(() -> zero(x), s.velocity, x)
    @. v = ρ * v + η * dx
    @. x -= v
end

mutable struct RAdam <: AbstractStep
    η
    β
    ϵ
    state
end

RAdam(; η = 0.001, β = (0.9, 0.999), ϵ = 1e-8) = RAdam(η, β, ϵ, IdDict())

function step!(o::RAdam, x, dx)
    (; η, β, ϵ) = o
    ρinf = 2 / (1-β[2]) - 1

    mt, vt, βp, t = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]], Ref(1))
    end

    Δ = dx

    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    ρ = ρinf - 2t[] * βp[2] / (1 - βp[2])
    if ρ > 4
        r = sqrt((ρ-4)*(ρ-2)*ρinf/((ρinf-4)*(ρinf-2)*ρ))
        @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η * r
    else
        @. Δ =  mt / (1 - βp[1]) * η
    end
    βp .= βp .* β
    t[] += 1

    @. x -= Δ
end

mutable struct AMSGrad <: AbstractStep 
    eta::Float64
    beta::Tuple{Float64, Float64}
    epsilon::Float64
    state::IdDict{Any, Any}
end

AMSGrad(η::Real = 0.001, β = (0.9, 0.999), ϵ::Real = EPS) = AMSGrad(η, β, ϵ, IdDict())
AMSGrad(η::Real, β::Tuple, state::IdDict) = AMSGrad(η, β, EPS, state)

function step!(o::AMSGrad, x, Δ)
  η, β = o.eta, o.beta

  mt, vt, v̂t = get!(o.state, x) do
    (fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon))
  end 

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  @. Δ = η * mt / (√v̂t + o.epsilon)

  @. x -= Δ
end
