using Test
using CUDAnative
using CUDAdrv

include("../src/reduce.jl")



cols = 3200
rows = 700

input = ones(Float64, rows, cols)
output = zeros(Float64, 1, cols)

cpu_result = sum(input, dims=1)

gpu_input = CuArray(input)
gpu_output = CuArray(output)

gpu_reduce(gpu_input, gpu_output, rows, cols)

gpu_result = Array(gpu_output)

@assert cpu_result == gpu_result



