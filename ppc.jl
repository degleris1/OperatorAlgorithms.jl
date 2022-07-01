using BenchmarkTools
using CUDA
using Tullio
using LoopVectorization

# Test data
DATA = [
    0 8 2
    1 0 9
    4 5 0
]

function shortcut_cpu0!(R, D)
    n = size(D, 1)

    D_T = Matrix(adjoint(D))

    Threads.@threads for j in 1:n
        @inbounds for i in 1:n
            
            # Fill in R[i, j]
            v = 1000f0
            @turbo for k = 1:n
                v = min(v, D_T[k, i] + D[k, j])
            end
            
            R[i, j] = v
        end
    end
end

function shortcut_cpu1!(R, D)
    @tullio (min) R[i, j] = D[i, k] + D[k, j]
end

min1(x, y) = ifelse(x < y, x, y)

function kernel0!(R, D)
    n = size(D, 1)

    x = threadIdx().x + (blockIdx().x - 1)*blockDim().x
    y = threadIdx().y + (blockIdx().y - 1)*blockDim().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().x

    for j in x:stride_x:n
        @inbounds for i in x:stride_x:n
            # Fill in R[i, j]
            
            v = 1000f0
            for k in 1:n
                v = min(v, D[i, k] + D[k, j])
            end
            
            R[i, j] = v
        end
    end
    return 
end

function shortcut_gpu0!(R, D)
    N = size(D, 2)
    nb = ceil(Int, N/16)
    CUDA.@sync begin
        @cuda threads=(16, 16) blocks=(nb, nb) kernel0!(R, D)
    end
end

function shortcut_gpu1!(R, D)
    N = size(D, 2)

    kernel = @cuda launch=false kernel0!(R, D)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    CUDA.@sync begin
        kernel(R, D; threads, blocks)
    end
end




# =====
# Verify Correctness
# ====
# TODO





# =====
# Benchmark Code
# =====

N = 1000
FLOPS = N^3 * 2
tm(b) = BenchmarkTools.mean(b).time
gflops(b) = FLOPS / (tm(b) / 1e9) / 1e9

# CPU
APU_PER_CORE = 16
cpu_best = APU_PER_CORE * Threads.nthreads() * 2 * 2.3e9 / 1e9
println("Theoretical CPU Performance: $cpu_best gflops")

D = rand(1f0:10f0, N, N)
R = zeros(Float32, N, N)

b_cpu0 = @benchmark shortcut_cpu0!($R, $D) seconds=1
@show gflops(b_cpu0)
b_cpu1 = @benchmark shortcut_cpu1!($R, $D) seconds=1
@show gflops(b_cpu1)


# GPU
println()
gpu_best = 5120 * 2 * 1.1e9 / 1e9
println("Theoretical GPU Performance: $gpu_best gflops")

Dg = CuArray(D)
Rg = CuArray{Float32}(undef, N, N)

b_gpu0 = @benchmark shortcut_gpu0!($Rg, $Dg) seconds=1
@show gflops(b_gpu0)
b_gpu1 = @benchmark shortcut_gpu1!($Rg, $Dg) seconds=1
@show gflops(b_gpu2)

# OUTPUT
# ======
# Theoretical CPU Performance: 294.4 gflops
# gflops(b_cpu0) = 41.37827821157856
# gflops(b_cpu1) = 195.3559481658156

# Theoretical GPU Performance: 11264.0 gflops
# gflops(b_gpu0) = 522.092821605634
# gflops(b_gpu1) = 4241.6644833591245

;
