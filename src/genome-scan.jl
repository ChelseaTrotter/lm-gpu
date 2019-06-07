
# Date: Sep 17, 2018
# Authur: Chelsea Trotter
# This program tests how long does genome scan takes. 
# Currently I am testing only square matrix, (r >= m >= n). In the future, I may test whether the shape of the matrix affects computation time.



include("env.jl")
include("util.jl")
include("../../src/benchmark.jl")
using Distributed
using DelimitedFiles
#import CuArrays.CuArray
import Base.@elapsed 



#n: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 
#m:                                            25600, 
#r:                                                , 51200, 102400, 204800, 409600, 819200, 1638400


function calculate_px(x::Array{Float64,2})
    XtX = transpose(x)*x
    result = X*inv(XtX)*transpose(x)
    # display(result)
    return result
end

function process_x(x)
    # # Random create covariates for now. This for loop can be used for real data later. 
    # for row in 1:size(x)[1] 
    #     if x[row] == "f"
    #         x[row] = 1.0
    #     else
    #         x[row] = -1.0
    #     end
    # end
    # return convert(Array{Float64,1}, x)

    return rand([-1.0,1.0], size(x)[1])
end

function get_standardized_matrix(m)
    for col in 1:size(m)[2]
        summ::Float64 = 0.0
        rows = size(m)[1]
        for row in 1:rows
            summ += m[row, col]
        end
        mean = summ/rows
        sums::Float64 = 0.0
        for row in 1:rows
            sums += (m[row,col] - mean)^2
        end
        std = sqrt(sums/rows)
        for row in 1:rows 
            m[row,col] = (m[row,col]-mean)/std   
        end
    end
    return m

end


function calculate_r(a::Array,b::Array)
    return LinearAlgebra.BLAS.gemm('T', 'N', a,b);
end

function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.CUBLAS.gemm('T', 'N', a,b);
end   

function lod_score_multithread(m,r::Array{Float64,2})
    n = convert(Float64,m)
    Threads.@threads for j in 1:size(r)[2]
        for i in 1:size(r)[1]
            r_square::Float64 = (r[i,j]/n)^2
            r[i,j]= -n/Float64(2.0) * log(Float64(1.0)-r_square)
        end
    end
    return r
end

function lod_score(n, r::Array{Float64,2})
    for j in 1:size(r)[2]
        for i in 1:size(r)[1]
            # print("$i,$j: $(r[i,j]); ")
            r_square::Float64 = (r[i,j]/n)^2
            r[i,j] = -n/Float64(2.0) * log(Float64(1.0)-r_square)
        end
    end
        return r
end

function my_isapprox(x,y)
    return isapprox(x,y, atol=1e-3);
end

function check_correctness(a, b)
        
    if(all(map(my_isapprox, a, b)))
        return "true"
    else
        return "false"
    end
end

function cpurun_with_covar(Y::Array{Float64,2}, G::Array{Float64,2}, X::Array{Float64,2}, n)
    px = calculate_px(X)
    # display(px)
    y_hat = LinearAlgebra.BLAS.gemm('N', 'N', px, Y)
    g_hat = LinearAlgebra.BLAS.gemm('N', 'N', px, G)
    y_tilda = Y .- y_hat
    g_tilda = G .- g_hat
    y_std = get_standardized_matrix(y_tilda)
    g_std = get_standardized_matrix(g_tilda)
    r = calculate_r(y_std, g_std)
    lod = lod_score_multithread(n, r)
    return lod
end

function cpurun(a::Array, b::Array, n)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    #step 2: calculate R, matrix of corelation coefficients
    r = calculate_r(a_std,b_std);
    #step 3: calculate r square and lod score
    # lod = lod_score(n, r);
    lod = lod_score_multithread(n,r)
    return lod
end

function gpurun(a::Array{Float64,2}, b::Array{Float64,2},n,m,p)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    d_a = CuArray(a_std);
    d_b = CuArray(b_std);
    d_r = calculate_r(d_a,d_b);
    gpu_square_lod(d_r,n,m,p)
    return collect(d_r)
end

function gpu_square_lod(d_r::CuArray{Float64,2},n,m,p)
    #Get total number of threads 
    ndrange = prod(size(d_r))
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
    # result = CuArray(zeros(m,p))
    blocks = min(Int(ceil(ndrange/threads)), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    return @cuda blocks=blocks threads=threads lod_kernel(d_r, ndrange,n)
     
end

function lod_kernel(input, MAX,n) 
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i < MAX+1)
        # r_square = (a[i] * a[i]) / (n*n)
        r_square = (input[i]/n)^2
        input[i] = (-n/Float64(2.0)) * CUDAnative.log(Float64(1.0)-r_square)
    end 
    return 
end

function print_cpu_timing(a, b, n)
    a_std = time_me_with_return(get_standardized_matrix,a);
    b_std = time_me_with_return(get_standardized_matrix,b);
    #step 2: calculate R, matrix of corelation coefficients
    r = time_me_with_return(calculate_r,a_std,b_std);
    #step 3: calculate r square and lod score
    lod = time_me_with_return(lod_score_multithread,n, r);
    return lod

end

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

            Y = convert(Array{Float64,2},readdlm("/nics/d/home/xiaoqihu/hg/fastlmm-gpu/bxdData/traits.csv", ','; skipstart=1)[:,2:end-1])

            G_prob = convert(Array{Float64,2},readdlm("/nics/d/home/xiaoqihu/hg/fastlmm-gpu/bxdData/geno_prob.csv", ','; skipstart=1)[:,2:end])

            G = G_prob[:, 1:2:end]

            X_temp = readdlm("/nics/d/home/xiaoqihu/hg/fastlmm-gpu/bxdData/traits.csv", ','; skipstart=1)[:,end]
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

                # cpu_result = benchmark(10, cpurun, Y, G,n);
                # gpu_result = benchmark(10, gpurun, Y, G,n,m,p);
                # speedup = cpu_result[3]/gpu_result[3];

                # println("$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");


                cpurun_with_covar(Y, G, X, n)
                cpu_result = benchmark(10,cpurun, Y, G, n)
                cpu_covar_result = benchmark(10, cpurun_with_covar, Y,G,X,n)
                println("$m, $n, $p, $(cpu_result[3]),  $(cpu_covar_result[3])\n");

                # println("Correctnes result : ", check_correctness(cpurun(Y,G,n), cpurun_with_covar(Y,G,X,n)))


                # println("std_time for Y is $std_time")
                # write(file, "$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
                # close(file);
            end
#         end
#     end

# end

# close(file)






