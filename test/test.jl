
using Distributed
using DelimitedFiles
using BenchmarkTools
using Dates
using Random
using Statistics
using LinearAlgebra
using CuArrays
using CUDAnative

import Base.@elapsed 


include("../src/util.jl")
include("../src/cpu.jl")
include("../src/gpu.jl")
include("../src/common.jl")



dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS");
host = gethostname();

# file = open("../timing/genome-scan-timing@$host@$dt_now.csv", "w");
# n_range = [1000]
# m_range = [20000, 40000]
# p_range = [1600,3200,6400,12800,25600, 51200, 102400, 204800, 409600, 819200, 1638400]

LinearAlgebra.BLAS.set_num_threads(16)
core_nums = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
println("Number of threads using: $core_nums")


# for n in n_range
#     for m in m_range    
#         for p in p_range

            Y = convert(Array{Float64,2},readdlm("../data/traits.csv", ','; skipstart=1)[:,2:end-1])

            G_prob = convert(Array{Float64,2},readdlm("../data/geno_prob.csv", ','; skipstart=1)[:,2:end])

            G = G_prob[:, 1:2:end]

            X_temp = readdlm("../data/traits.csv", ','; skipstart=1)[:,end]
            # println("size of X: $(size(X_temp)[1]), $(typeof(X_temp))")
            
            X = hcat(ones(size(X_temp)[1]),process_x(X_temp))
            # X = ones(size(X_temp)[1])

            n = size(Y,1)
            m = size(Y,2)
            p = size(G,2)

            println("*************************** n: $n,m: $m, p: $p******************************");

            if ((n*m + n*p + m*p) > get_max_doubles())
                println("too big to process")
                # file = open("../timing/genome-scan-timing@$host@$dt_now.csv", "a");
                # println("Matrices are too big to fit in GPU memory. Skipping this configuration.  N is $n, M is $m, P is $p");
                # write(file, "Matrices are too big to fit in GPU memory. Skipping this configuration.  N is $n, M is $m, P is $p\n");
                # close(file);
            else
                # file = open("../timing/genome-scan-timing@$host@$dt_now.csv", "a");
                
                
                # srand(123);

                # Y = rand(n, m); # full rank matrix 
                # G = rand([-1.0,0.0,1.0],(n, p)); # (-1,0,1) integers, or real numbers between (0,1), may be or not be full rank

                # print_cpu_timing(Y,G,n)
                # time_me_with_return(cpurun, Y,G,n);
                # time_me_with_return(gpurun, Y,G,n,m,p);

                cpu_result = benchmark(10, cpurun, Y, G,n);
                gpu_result = benchmark(10, gpurun, Y, G,n,m,p);
                speedup = cpu_result[3]/gpu_result[3];

                println("$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");


                # cpurun_with_covar(Y, G, X, n)
                # cpu_result = benchmark(10,cpurun, Y, G, n)
                # cpu_covar_result = benchmark(10, cpurun_with_covar, Y,G,X,n)
                # println("$m, $n, $p, $(cpu_result[3]),  $(cpu_covar_result[3])\n");

                # println("Correctnes result : ", check_correctness(cpurun(Y,G,n), cpurun_with_covar(Y,G,X,n)))


                # println("std_time for Y is $std_time")
                # write(file, "$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
                # close(file);
            end
#         end
#     end

# end

# close(file)