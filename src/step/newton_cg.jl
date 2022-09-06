
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

        _x = zero(b1)
        x_buf = zero(b1)
        y_buf = zero(b2)

        K! = (w, y) -> begin  # K = Q' * inv(H) * Q x
            # _x = Qx!(_x, F, y, x_buf, y_buf)
            # @. _x /= H.diag
            # w = Qtx!(w, F, _x, x_buf, y_buf)
            w .= Qtx(F, Qx(F, y) ./ H.diag)
            return w
        end

        u, cnt, cg_error = solve_cg!(u0, K!, b̂, num_iter)

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

    return dz, cnt, cg_error
end

# TODO: Cache r, p, w
function solve_cg!(u, K!, b, num_iter; atol=1e-10, rtol=1e-3)

    u .= b  # The gradient direction is a good initial guess at the Newton direction
    r = b - K!(zero(u), u)
    
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

        w = K!(w, p)  # w .= K(p)
        α = rho[iter] / (p'w)
        
        @. u = u + α*p
        @. r = r - α*w
        
        push!(rho, norm(r)^2)
        
        @. p = r + (rho[iter+1] / rho[iter]) * p
    end
    # @show cnt, norm_b, sqrt(rho[end])

    return u, cnt, sqrt(rho[end]) / norm_b
end

# ====
# SPARSE QR OPERATIONS
# ====

function Rx(F, x)
    @inbounds return F.R * x[F.pcol]
end

function Rtx(F, x)
    @inbounds return (F.Rt * x)[F.ipcol]
end

function Qx(F, x)
    @inbounds return (F.Q * x)[F.iprow]
end

function Qx!(z, F, y, x_buf, y_buf)
    mul!(x_buf, F.Q, y)
    @inbounds z .= x_buf[F.iprow]  # OPTIMIZE allocation
    return z
end

function Qtx(F, x)
    @inbounds return F.Q' * x[F.prow]
end

function Qtx!(z, F, x, x_buf, y_buf)
    @inbounds x_buf .= x[F.prow]  # OPTIMIZE allocation
    mul!(z, F.Q', x_buf)
    return z
end

function R_div_b(F, b)
    # (F.R \ b)[F.ipcol)
    y = zero(b)
    ldiv!(y, UpperTriangular(F.R), b)
    @inbounds return y[F.ipcol]
end

function Rt_div_b(F, b)
    # F.Rt \ b[F.pcol]
    y = zero(b)
    ldiv!(y, LowerTriangular(F.Rt), b[F.pcol])
    @inbounds return y
end
