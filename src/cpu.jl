
function calculate_r(a::Array,b::Array)
    return LinearAlgebra.BLAS.gemm('T', 'N', a,b);
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

function print_cpu_timing(a, b, n)
    a_std = time_me_with_return(get_standardized_matrix,a);
    b_std = time_me_with_return(get_standardized_matrix,b);
    #step 2: calculate R, matrix of corelation coefficients
    r = time_me_with_return(calculate_r,a_std,b_std);
    #step 3: calculate r square and lod score
    lod = time_me_with_return(lod_score_multithread,n, r);
    return lod

end
