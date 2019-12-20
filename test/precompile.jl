# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

# +
using SnoopCompile

SnoopCompile.@snoopc "/tmp/lm_gpu_compiles.log" begin 
    include("test.jl")
end

packages = SnoopCompile.read("/tmp/lm_gpu_compiles.log")
parcels = SnoopCompile.parcel(reverse!(packages[2]))
SnoopCompile.write("/tmp/precompile", parcels)
# -

?mktemp

# +

function cp_precompiled_file(packageName::String)
    precompile_file = "precompile_" * packageName * ".jl"
    dir = dirname(Base.find_package(packageName))
    Base.Filesystem.cp("/tmp/precompile/$precompile_file", "$dir/precompile.jl",force=true)
end

function add_precompile_to_package(packageName::String)
    mode = 666 # file access mode, 666 is read write for all user
    filename = Base.find_package(packageName)
    # check if precompile code is there
    (tmppath, tmpio) = mktemp()
    open(filename) do io 
        for line in eachline(io)
            
        
        
        
    # if not, insert it.
    Base.Filesystem.chmod(filename, mode)
    
    readline(filename)
end

function install_packages(parcels::Dict)
    for (pkgs, funcs) in parcels
        p = String(pkgs)
        if(haskey(Pkg.installed(), p))
            println("✅$p is Installed, now inserting precompiled function to $p")
            # add precompile functions to src code. 
            cp_precompiled_file(p)
        else
            println("$p is Not Installed")
            try
                println("Installing $p...")
                Pkg.add(p)
                cp_precompiled_file(p)
            catch e
                println("Can't install package $p !")
            end
        end    
         
    end
end

install_packages(parcels)

# Get a list of packages that has precompile functions in. 
# insert precompile file into each package src file.

# -

src_file = open(Base.find_package("Compat"))
seekend(src_file)
read(src_file, String)

test_file = open("./test.jl")
countlines(test_file)

using Pkg
haskey(Pkg.installed(),"TextWrap")

s = open("test.txt", "a+")
write(s, "B\n")
write(s, "C\n")
println(position(s))
seekend(s)
println(position(s))
write(s, "D\n")
close(s)

# find end of line
s = open(Base.find_package("Compat"), "r")
prevline = readline(s)
x = readline(s)
while !eof(s) 
#     println(x)
    prevline = x
    x = readline(s)
end
close(s)


x


