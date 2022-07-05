using BenchmarkTools
using CUDA
using Tullio
using LoopVectorization

# Test data
TEST_DATA = Float32.([
    0 8 2
    1 0 9
    4 5 0
])

EXPECTED_OUTPUT = Float32.([
    0 7 2
    1 0 3
    4 5 0
])

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

    # NOTE: try switching i and j and see what happens to performance
    # (Because of the memory access pattern, putting i in the outer loop is
    # significantly faster)
    for i in x:stride_x:n
        @inbounds for j in y:stride_y:n
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

function matmul(R, A, B)
    CUDA.@sync R .= A*B
end

function shortcut_gpu0!(R, D)
    NUM_THREADS = 16
    N = size(D, 2)
    nb = ceil(Int, N/NUM_THREADS)
    CUDA.@sync begin
        @cuda threads=(NUM_THREADS, NUM_THREADS) blocks=(nb, nb) kernel0!(R, D)
    end
end

function shortcut_gpu1!(R, D)
    N = size(D, 2)

    kernel = @cuda launch=false kernel0!(R, D)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    threads = floor(Int, sqrt(threads))
    threads = (threads, threads)
    blocks = floor(Int, sqrt(blocks))
    blocks = (blocks, blocks)

    CUDA.@sync begin
        kernel(R, D; threads, blocks)
    end
end




# =====
# Verify Correctness
# ====

function test_cpu_alg(alg)
    N = size(TEST_DATA, 2)
    R = zeros(Float32, N, N)

    alg(R, TEST_DATA)

    return R ≈ EXPECTED_OUTPUT
end

function test_gpu_alg(alg)
    N = size(TEST_DATA, 2)
    R = CuArray(zeros(Float32, N, N))

    alg(R, CuArray(TEST_DATA))

    return Array(R) ≈ EXPECTED_OUTPUT
end

@show test_cpu_alg(shortcut_cpu0!)
@show test_cpu_alg(shortcut_cpu1!)

@show test_gpu_alg(shortcut_gpu0!)
@show test_gpu_alg(shortcut_gpu1!)




# =====
# Benchmark Code
# =====

N = 1024
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
# b_gpu1 = @benchmark shortcut_gpu1!($Rg, $Dg) seconds=1
# @show gflops(b_gpu1)

Dg_T = CuArray(adjoint(D))
b_mat = @benchmark matmul($Rg, $Dg_T, $Dg)
@show gflops(b_mat)

;
