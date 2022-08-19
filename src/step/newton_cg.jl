
# TODO: Optimize
function solve_schur_cg!(dz, H, A; num_iter=10, qr_factors=nothing, u0=nothing)
    @assert all(
        H.diag .> sqrt(eps())
    ) "Hessian nearly singular: index $(argmin(H.diag)), value $(minimum(H.diag))"

    # @show extrema(H.diag), maximum(H.diag) / minimum(H.diag)

    dx, dy = dz.primal, dz.dual
    u0 = something(u0, zero(dy))

    # Create right hand size
    b1, b2 = copy(dx), copy(dy)

    # Eliminate upper block
    b̃ = A * (H \ b1) - b2

    # Solve subsystem without QR
    if isnothing(qr_factors)
        K = x -> A * (H \ (A' * x))
        dy .= solve_cg!(u0, K, b̃, num_iter)

    # Solve subsystem with QR
    else
        F = qr_factors

        b̂ = Rt_div_b(F, b̃)  # R' \ b̃
        K = x -> Qtx(F, (H \ Qx(F, x)))  # K = Q' H Q x

        u = solve_cg!(u0, K, b̂, num_iter)

        cg_error = norm(K(u) - b̂)
        if sqrt(cg_error) > 1e-1
            @warn "CG system only solved to: $(sqrt(cg_error)) accuracy"
        end


        dy .= R_div_b(F, u)  # R \ u
    end

    # Back solve
    dx .= H \ (b1 - A' * dy)

    # Check error
    # @show norm(A * inv(H) * A' * dy - b̃)
    # @show norm(H * dx + A' * dy - b1)
    # @show norm(A * dx - b2)

    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual

    return dz
end

# TODO: Cache r, p, w
function solve_cg!(u0, K, b, num_iter; ϵ=1e-8, upper_tol=1e-1)
    u = u0  # We override u0
    r = b - K(u0)
    
    p = copy(r)
    w = zero(b)

    rho = [norm(r)^2]
    for iter in 1:num_iter
        if sqrt(rho[iter]) < ϵ
            break
        end

        w .= K(p)
        α = rho[iter] / (p'w)
        
        @. u = u + α*p
        @. r = r - α*w
        
        push!(rho, norm(r)^2)
        
        @. p = r + (rho[iter+1] / rho[iter]) * p
    end

    return u
end

# ====
# SPARSE QR OPERATIONS
# (Since Julia sparse linear algebra is incomplete...)
# ====

function Rx(F, x)
    return F.R * x[F.pcol]
end

function Rtx(F, x)
    return (F.R' * x)[invperm(F.pcol)]
end

function Qx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return (F.Q * [x; zeros(m - n)])[invperm(F.prow)]
end

function Qtx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return (F.Q' * x[F.prow])[1:n]
end

function R_div_b(F, b)
    return (F.R \ b)[invperm(F.pcol)]
end

function Rt_div_b(F, b)
    return F.R' \ (b[F.pcol])
end