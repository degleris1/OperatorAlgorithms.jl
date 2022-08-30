
mutable struct PrimalDual{T <: Real, V <: AbstractVector{T}}
    primal::V
    dual::V
end

Base.zero(z::PrimalDual) = PrimalDual(zero(z.primal), zero(z.dual))

Base.similar(z::PrimalDual) = PrimalDual(similar(z.primal), similar(z.dual))

function update!(x::Array, x1::Array, t, dx::Array)
    @. x = x1 + t*dx
    return x
end

function update!(z::PrimalDual, z1::PrimalDual, t, dz::PrimalDual)
    x, y = z.primal, z.dual
    x1, y1 = z1.primal, z1.dual
    dx, dy = dz.primal, dz.dual

    @. x = x1 + t*dx
    @. y = y1 + t*dy

    return z
end

function norm(z::PrimalDual, p::Real=2)
    return (norm(z.primal, p)^p + norm(z.dual, p)^p) ^ (1/p)
end


# Base.ndims(::Type{PrimalDual}) = 1

# function Base.size(z::PrimalDual)
#     return (length(z), )
# end

# function Base.length(z::PrimalDual)
#     return length(z.primal)+length(z.dual)
# end

# function Base.getindex(z::PrimalDual, i)
#     if i <= length(z.primal)
#         return z.primal[i]
#     else
#         return z.dual[i-length(z.primal)]
#     end
# end

# function Base.iterate(z::PrimalDual, i=0)
#     if i < length(z)
#         return (z[i+1], i+1)
#     else
#         return nothing
#     end
# end

# function Base.copyto!(z::PrimalDual, z2)
#     
