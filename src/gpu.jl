
function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.BLAS.gemm('T', 'N', a,b);
end   

function gpurun(a::Array{Float64,2}, b::Array{Float64,2},n,m,p)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    d_a = CuArray(a_std);
    d_b = CuArray(b_std);
    d_output = CuArray(zeros(Float64, 1, p))
    d_r = calculate_r(d_a,d_b);
    gpu_square_lod(d_r,n)
    gpu_reduce(d_r,d_output, m,p)
    return collect(d_output)
end

function gpu_square_lod(d_r::CuArray{Float64,2},n)
    #Get total number of threads 
    ndrange = prod(size(d_r))
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
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

function reduce_kernel(input, output, ndrange, rows)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i < ndrange+1)
        for r in 1:rows
            output[i] += input[r]
        end
    end
end


function gpu_reduce(input::CuArray{Float64,2},output::CuArray{Float64,2}, rows, cols)
    #Get total number of threads 
    ndrange = cols
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
    # result = CuArray(zeros(m,p))
    # blocks = min(Int(ceil(ndrange/threads)), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    blocks = Int(ceil(ndrange/threads))
    return @cuda blocks=blocks threads=threads reduce_kernel(input,output,ndrange,rows)
     
end
