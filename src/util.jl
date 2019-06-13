

function my_isapprox(x,y)
    return isapprox(x,y, atol=1e-3);
end

function check_correctness(a, b, length)
    for i in 1:length
        if(!isapprox(a[i],b[i], atol=1e-3))
            return false
        end 
    end
    return true
end

function set_blas_threads()
    LinearAlgebra.BLAS.set_num_threads(16)
    core_nums = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    println("Number of threads using: $core_nums")
end

function get_timing_file_name()
    dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS");
    host = gethostname();
    # file = open("../timing/genome-scan-timing@$host@$dt_now.csv", "a");
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



