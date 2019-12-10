
using Distributed
using DelimitedFiles
using BenchmarkTools
using Dates
using Random
using Statistics
using LinearAlgebra
# 

import Base.@elapsed 


include("../src/util.jl")
include("../src/cpu.jl")
# include("../src/gpu.jl")
include("../src/common.jl")
include("../src/cli.jl")
include("../src/read_data.jl")

function run_genome_scan(geno_file, pheno_file, export_matrix)

    G = get_geno_data(geno_file)
    Y = get_pheno_data(pheno_file)
    # X_temp = readdlm("../data/hippocampus-pheno-nomissing.csv", ','; skipstart=1)[:,half_pheno_size]
    # X = hcat(ones(size(X_temp)[1]),process_x(X_temp))
    
    n = size(Y,1)
    m = size(Y,2)
    p = size(G,2)

    println("*************************** n: $n,m: $m, p: $p******************************");

    # cpu = cpurun(Y, G,n)
    # gpu = gpurun(Y,G,n,m,p)

    # compare_cpu_gpu_result(Y,G,n,m,p);
    # cpu_timing = benchmark(5, cpurun, Y, G,n);
    # gpu_timing = benchmark(5, gpurun, Y, G,n,m,p);
    # speedup = cpu_timing[3]/gpu_timing[3];

    # println("$m, $n, $p, $(cpu_timing[3]),  $(gpu_timing[3]), $speedup\n");

    cpurun(Y, G,n,export_matrix)

end

function compare_cpu_gpu_result(Y,G,n,m,p)
    cpu_result = cpurun(Y,G,n)
    gpu_result = gpurun(Y,G,n,m,p)

    # cpu_max = cpu_result[1] 
    # cpu_max_idx = Array{Int64}(undef, m)
    # for i in 1:m
    #     cpu_max_idx[i] = cpu_result[2][i][2]
    # end
    
    idx_match = check_correctness_idx(cpu_result[:,1], gpu_result[:,1])
    max_match = check_correctness(cpu_result[:,2], gpu_result[:,2])


    # idx_match = check_correctness(cpu_max_idx, gpu_result[:,1], cpu_max, gpu_result[:,2])
    # max_match = check_correctness(cpu_max, gpu_result[:,2])

    println("Correctness result: Index match? $idx_match, Max Match? $max_match.")

end

function run_simulation(n,m,p)
    
    n = 70
    m = 700
    p = 3200

    Y = rand(n, m); # full rank matrix 
    G = rand([-1.0,0.0,1.0],(n, p)); # (-1,0,1) integers, or real numbers between (0,1), may be or not be full rank

    println("*************************** n: $n,m: $m, p: $p******************************");

    cpu_result = benchmark(1, cpurun, Y, G,n);
    gpu_result = benchmark(1, gpurun, Y, G,n,m,p);
    speedup = cpu_result[3]/gpu_result[3];

    println("$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
end


function main()
    cli_args = parse_commandline()
    geno_file = cli_args["geno_file"]
    pheno_file = cli_args["pheno_file"]
    export_matrix = cli_args["export_matrix"]
    set_blas_threads();
    run_genome_scan(geno_file, pheno_file, export_matrix);

    # # picking threashold of rowsum of NA
    # th = cli_args["threashold"]
    # # use pseudomarker to impute genotype
    # use_pseudomarker = cli_args["use_pseudomarker"]
    # # seed for random number generator to impute phenotype
    # seed = cli_args["rseed"]
    
end

main() 


# run_simulation(n,m,p);





