
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
        cpu_result = benchmark(10, cpurun, Y, G,n);
        gpu_result = benchmark(10, gpurun, Y, G,n,m,p);
        speedup = cpu_result[3]/gpu_result[3];

        println("$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    end

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
run_simulation(n,m,p);




