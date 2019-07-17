

function reduce_kernel(input, output, ndrange, rows)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i < ndrange+1)
        for r in 1:rows
            output[i] += input[r]
        end
    end
end

function gpu_reduce(input::CuArray{Float32,2},output::CuArray{Float32,2}, rows, cols)
    #Get total number of threads 
    ndrange = cols
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
    # result = CuArray(zeros(m,p))
    blocks = min(Int(ceil(ndrange/threads)), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    return @cuda blocks=blocks threads=threads reduce_kernel(input,output,ndrange,rows)
     
end

