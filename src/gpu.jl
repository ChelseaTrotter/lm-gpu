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
    # gpu_reduce(d_r,m,p)
    return collect(d_r[:, 1:3])
end

function gpu_square_lod(d_r::CuArray{Float64,2},n,m,p)
    #Get total number of threads 
    ndrange = prod(size(d_r))
    # println("Size of d_r $(size(d_r))")
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
        # r_square = (a[i] * a[i]) / (n*n)
        r_square = (input[tid]/n)^2
        input[tid] = (-n/Float64(2.0)) * CUDAnative.log(Float64(1.0)-r_square)
    end 
    return
end

function reduce_kernel(input, rows, cols)
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # Trying for shared memory:
    # shmem = @cuDynamicSharedMem(Float64, rows)
    # for i in 1:rows
    #     shmem[i] = input[i, tid]
    # end
    # temp_max = shmem[1]

    # Trying for simplest kernel
    if(tid < rows+1)
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
    if(tid<cols)
        input[tid,3] = input[79,tid]
    end
    # if(tid < rows+1)
    #     for i in 1:cols
    #         input[tid, 1] = i
    #     end
    # end

    return
end

# function gpu_reduce(input::CuArray{Float64,2}, rows, cols)
    #Get total number of threads 
    # ndrange = cols
    #Get maximum number of threads per block
    # dev = device()
    # threads = attribute(dev, CUDAdrv.WARP_SIZE)
    # result = CuArray(zeros(m,p))
    # blocks = min(Int(ceil(ndrange/threads)), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    # blocks = Int(ceil(ndrange/threads))

    # Trying for shared memory. 
    # shmem = cols * sizeof(Float64)
    # return @cuda blocks=1 threads=cols shmem=shmem reduce_kernel(input,rows,cols)

#     return @cuda blocks=1 threads=cols reduce_kernel(input,rows,cols)
     
# end
