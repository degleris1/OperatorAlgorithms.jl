
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

        u, cnt, cg_error = solve_cg!(u0, K, b̃, num_iter)
        dy .= u

    # Solve subsystem with QR
    else
        F = qr_factors

        b̂ = Rt_div_b(F, b̃)  # R' \ b̃

        K = x -> Qtx(F, (H \ Qx(F, x)))  # K = Q' H Q x
        u, cnt, cg_error = solve_cg!(u0, K, b̂, num_iter)

        dy .= R_div_b(F, u)  # R \ u

    end

    if cg_error > 1e-2
        @warn "CG system only solved to: $(sqrt(cg_error)) accuracy"
    end

    # Back solve
    dx .= H \ (b1 - A' * dy)

    # Check error
    # @show norm(A * inv(H) * A' * dy - b̃)
    # @show norm(H * dx + A' * dy - b1)
    # @show norm(A * dx - b2)

    @. dz.primal = -dz.primal
    @. dz.dual = -dz.dual

    return dz, cnt, cg_error
end

# TODO: Cache r, p, w
function solve_cg!(u, K, b, num_iter; atol=1e-10, rtol=1e-4)

    u .= b  # The gradient direction is a good initial guess at the Newton direction
    r = b - K(u)
    
    p = copy(r)
    w = zero(b)

    rho = [norm(r)^2]
    cnt = 0
    norm_b = norm(b)

    for iter in 1:num_iter
        if sqrt(rho[iter]) < atol + rtol*norm_b
            break
        end

        cnt += 1

        w .= K(p)
        α = rho[iter] / (p'w)
        
        @. u = u + α*p
        @. r = r - α*w
        
        push!(rho, norm(r)^2)
        
        @. p = r + (rho[iter+1] / rho[iter]) * p
    end
    @show cnt, norm_b, sqrt(rho[end])

    return u, cnt, sqrt(rho[end]) / norm_b
end

# ====
# SPARSE QR OPERATIONS
# (Since Julia sparse linear algebra is incomplete...)
# ====

function Rx(F, x)
    return F.R * x[F.pcol]
end

function Rtx(F, x)
    return (F.Rt * x)[invperm(F.pcol)]
end

function Qx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return (F.Q * x)[invperm(F.prow)]
end

function Qtx(F, x)
    m, n = length(F.prow), length(F.pcol)
    return F.Q' * x[F.prow]
end

function R_div_b(F, b)
    return (F.R \ b)[invperm(F.pcol)]
end

function Rt_div_b(F, b)
    return F.Rt \ b[F.pcol]
end
