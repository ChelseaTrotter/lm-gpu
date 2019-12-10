using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        # "url"
        #     help = "A url that points to the data location"
        #     required = true
        "--geno_file"
            help = "input geno type file"
            required = true
        "--pheno_file"
            help = "clean pheno type file"
            required = true
        "--export_matrix"
            help = "if flaged, lod matrix will be exported. If not flagged, a vector of max lod will be exported."
            action = :store_true
        "--output_file"
            help = "name of the file you want to output"
            default = "output.csv"

        # "--threashold"
        #     help = "Threashold of NA to keep individuals"
        #     arg_type = Int
        #     default = 10    # 0 means no missing data, therefore no imputation. 
        #                     # 1-10 means there are missing data, need to impute and provide a seed to the following argument. 
        # "--rseed"
        #     help = "Provide a seed for randome number generator"
        #     default = 300
        # "--use_pseudomarker"
        #     help = "Use pseudomarker or not"
        #     action = :store_true
    end

    return parse_args(s)
end
