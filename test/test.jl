
using Distributed
using DelimitedFiles
using BenchmarkTools
using Dates
using Random
using Statistics
using LinearAlgebra
using CuArrays
using CUDAnative
using CUDAdrv

import Base.@elapsed 


include("../src/util.jl")
include("../src/cpu.jl")
include("../src/gpu.jl")
include("../src/common.jl")

function run_genome_scan()
    Y = convert(Array{Float64,2},readdlm("../data/traits.csv", ','; skipstart=1)[:,2:end-1])
    G_prob = convert(Array{Float64,2},readdlm("../data/geno_prob.csv", ','; skipstart=1)[:,2:end])
    G = G_prob[:, 1:2:end]
    X_temp = readdlm("../data/traits.csv", ','; skipstart=1)[:,end]
    X = hcat(ones(size(X_temp)[1]),process_x(X_temp))

    n = size(Y,1)
    m = size(Y,2)
    p = size(G,2)

    println("*************************** n: $n,m: $m, p: $p******************************");

    if ((n*m + n*p + m*p) > get_max_doubles())
        println("too big to process")
    else

        # check_gpu_idx(Y,G,n,m,p)
        # check_cpu_idx(Y,G,n)
        compare_cpu_gpu_result(Y,G,n,m,p);
        # cpu_timing = benchmark(10, cpurun, Y, G,n);
        # gpu_timing = benchmark(10, gpurun, Y, G,n,m,p);
        # speedup = cpu_timing[3]/gpu_timing[3];

        # println("$m, $n, $p, $(cpu_timing[3]),  $(gpu_timing[3]), $speedup\n");
    end

end

function compare_cpu_gpu_result(Y,G,n,m,p)
    cpu_result = cpurun(Y,G,n)
    gpu_result = gpurun(Y,G,n,m,p)

    cpu_max = cpu_result[1] 
    cpu_max_idx = Array{Int64}(undef, m)
    # println(cpu_result)
    for i in 1:m
        # print(cpu_result[2][i])
        # print("$(cpu_result[2][i][2]), ")
        cpu_max_idx[i] = cpu_result[2][i][2]
    end

    # idx_match = cpu_max_idx ≈ gpu_result[:,1]
    # max_match = cpu_max ≈ gpu_result[:,2]

    idx_match = check_correctness(cpu_max_idx[70:80], gpu_result[70:80,1])
    max_match = check_correctness(cpu_max, gpu_result[:,2])

    # println("GPU lod row 79:")
    # display(gpu_result[1:7321, 3])
    open("row79gpu.csv", "w") do io 
        writedlm(io, gpu_result[1:7321, 3])
    end
    
    # Printing idx matching result
    # display(cpu_result[2][1:10])
    # display(cpu_max_idx[70:80])
    # display(gpu_result[70:80,1])

    # Printing max of each column result
    # display(cpu_max[35550:35556])
    # display(gpu_result[35550:35556, 2])

    # println(size(cpu_max))
    # println(size(gpu_result))

    println("Correctness result: Index match? $idx_match, Max Match? $max_match.")

end

function check_gpu_idx(Y,G,n,m,p)
    gpu_result = gpurun(Y,G,n,m,p)
    display(gpu_result[1:90, 1])
end

function check_cpu_idx(Y,G,n)
    cpu_result = cpurun(Y,G,n)
    display(cpu_result[1][])
end

function run_simulation(n,m,p)
    # srand(123);

    Y = rand(n, m); # full rank matrix 
    G = rand([-1.0,0.0,1.0],(n, p)); # (-1,0,1) integers, or real numbers between (0,1), may be or not be full rank

    println("*************************** n: $n,m: $m, p: $p******************************");

    cpu_result = benchmark(10, cpurun, Y, G,n);
    gpu_result = benchmark(10, gpurun, Y, G,n,m,p);
    speedup = cpu_result[3]/gpu_result[3];

    println("$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
end




n = 70
m = 700
p = 3200

set_blas_threads();
# run_simulation(n,m,p);
run_genome_scan();




