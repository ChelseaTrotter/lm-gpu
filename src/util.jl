

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

function print_cpu_timing(a, b, n)
    a_std = time_me_with_return(get_standardized_matrix,a);
    b_std = time_me_with_return(get_standardized_matrix,b);
    #step 2: calculate R, matrix of corelation coefficients
    r = time_me_with_return(calculate_r,a_std,b_std);
    #step 3: calculate r square and lod score
    lod = time_me_with_return(lod_score_multithread,n, r);
    return lod

end

function get_max_doubles()
    one_gb = 2^30
    acf_mem = 16*one_gb

    # println("Total GPU memory size: $gpu_mem_size bytes. \n")
    return (acf_mem/sizeof(Float64)) * 0.90
end

function get_max_singles()
    # one_hundred_MB = 134217728
    one_gb = 2^30
    my_mem = 16*one_gb
    # my_mem = get_gpu_mem_size()

    # println("Total GPU memory size: $gpu_mem_size bytes. Maximum Singles precision float is $(gpu_mem_size/size_of_single_float) \n")
    return (my_mem/sizeof(Float32)) * 0.90
end

function get_gpu_mem_size()
    dev = CuDevice(0)
    ctx = CuContext(dev)    
    mem = Mem.total()
    destroy!(ctx)
    return mem

end

function time_me_with_return(f::Function,x...)

    local returnvalue
    start = time_ns();
    returnvalue = f(x...)
    timing = Int((time_ns() - start)) / 1e+9
    println("Function $(string(f)) took $timing seconds. ")
    return returnvalue
end

function benchmark(nrep::Int64, f::Function,x...; result::Bool=false)

    res = Array{Float64}(undef, nrep)

    for i=1:nrep
        start = time_ns();
        f(x...)
        res[i] = Int((time_ns() - start)) / 1e+9
    end
    if(result)
        return (res)
    else
        return ([minimum(res) quantile(res,[0.25  0.5 0.75]) maximum(res)])
    end
end

function benchmarkWIthReturnValue(nrep::Int64, f::Function,x...; result::Bool=false)

    res = Array{Float64}(undef, nrep)
    local returnvalue
    for i=1:nrep
        tic()
        returnvalue = f(x...)
        res[i] = toq()
    end
    if(result)
        return (returnvalue,res)
    else
        return (returnvalue, [minimum(res) quantile(res,[0.25  0.5 0.75]) maximum(res)])
    end
end



