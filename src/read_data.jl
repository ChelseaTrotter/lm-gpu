using RCall
using DelimitedFiles

include("cli.jl")


function rcall_read_cross2()
    # url = ARGS
    url = "/Users/xiaoqihu/Documents/hg/lm-gpu/data/spleen/bxd-spleen.yaml"
    # url = "http://gn2-zach.genenetwork.org/api/v_pre1/genotypes/rqtl2/BXD.zip"

    R"""
    library(qtl2); 
    get_data <- function(url){
    data <- read_cross2(url)
    return(data)
    }
    """
    data = R"get_data"(url)
end

function get_geno_data(file)
    # The following two commnet lines shows how genotype file is processed originally in test.jl. 
    # G_prob = convert(Array{Float32,2},readdlm("../data/hippocampus-genopr-AA-BB.csv", ','; skipstart=1)[:,2:end])
    # G = G_prob[:, 1:2:end]
    
    geno_prob = convert(Array{Float32,2},readdlm(file, ','; skipstart=1)[:,2:end])
    println("got geno file.")
    return geno_prob[:,1:2:end]
end

function get_pheno_data(file)
    # The following two commnet lines shows how phenotype file is processed originally in test.jl. 
    # pheno = readdlm("../data/hippocampus-pheno-nomissing.csv", ','; skipstart=1)[:,2:end-1]
    # Y = convert(Array{Float32,2}, pheno[:, 1:end])

    pheno = readdlm(file, ','; skipstart=1)[:,2:end-1]
    println("got pheno file")
    return convert(Array{Float32,2}, pheno[:, 1:end])

end


