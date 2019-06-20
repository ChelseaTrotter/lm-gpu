using Test

# function my_isapprox(x,y)
#     return isapprox(x,y, atol=1e-3);
# end

function check_correctness(idx_a, idx_b, v_a, v_b)
    if (length(idx_a)!=length(idx_b))
        return println("Dimention mismatch when checking for correctness, abort checking.")
    end

    result = Array{Bool}(undef, length(idx_a))
    for i in 1:length(idx_a)
        b = convert(Int, idx_b[i])
        if(!isapprox(idx_a[i],b, atol=1e-3))
            # println("Mismatch at index: $i, comparing $(idx_a[i]) and $(b), 
            #         value is $(v_a[i]) and $(v_b[i]).")
            result[i] = false
        else
            result[i] = true
        end 

    end

    if(all(result))
        return true
    else 
        return false
    end
end

function check_correctness(v_a, v_b)
    if (length(v_a)!=length(v_b))
        return println("Dimention mismatch when checking for correctness, abort checking.")
    end

    result = Array{Bool}(undef, length(v_a))
    for i in 1:length(v_a)
        if(!isapprox(v_a[i],v_b[i], atol=1e-3))
            result[i] = false
        else
            result[i] = true
        end 
    end

    if(all(result))
        return true
    else 
        return false
    end
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



