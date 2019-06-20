import CuArrays.CuArray

function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.BLAS.gemm('T', 'N', a,b);
end   

function gpurun(a::Array{Float64,2}, b::Array{Float64,2},n,m,p)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    d_a = CuArray(a_std);
    d_b = CuArray(b_std);
    d_r = calculate_r(d_a,d_b);
    gpu_square_lod(d_r,n,m,p)
    return collect(d_r[:, 1:2])
end

function gpu_square_lod(d_r::CuArray{Float64,2},n,m,p)
    #Get total number of threads 
    ndrange = prod(size(d_r))
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
    blocks = min(Int(ceil(ndrange/threads)), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    @cuda blocks=blocks threads=threads lod_kernel(d_r, ndrange,n)
    return @cuda blocks=blocks threads=threads reduce_kernel(d_r,m,p)
    
end

function lod_kernel(input, MAX,n) 
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(tid < MAX+1)
        r_square = (input[tid]/n)^2
        input[tid] = (-n/Float64(2.0)) * CUDAnative.log(Float64(1.0)-r_square)
    end 
    return
end

function reduce_kernel(input, rows, cols)
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # Trying for simplest kernel
    if(tid <= rows)
        temp_max = input[tid, 1]
        max_idx = 1
        for i in 1:cols
            if temp_max < input[tid,i]
                temp_max = input[tid,i]
                max_idx = i
            end
        end
        input[tid,1] = max_idx
        input[tid,2] = temp_max
    end

    return
end
