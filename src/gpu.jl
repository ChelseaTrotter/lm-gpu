import CuArrays.CuArray

function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.CUBLAS.gemm('T', 'N', a,b);
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

