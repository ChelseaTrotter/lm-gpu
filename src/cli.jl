using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--threashold"
            help = "Threashold of NA to keep individuals"
            arg_type = Int
            default = 10    # 0 means no missing data, therefore no imputation. 
                            # 1-10 means there are missing data, need to impute and provide a seed to the following argument. 
        "--rseed"
            help = "Provide a seed for randome number generator"
            default = 300
        "--use_pseudomarker"
            help = "Use pseudomarker or not"
            action = :store_true
        "url"
            help = "A url that points to the data location"
            required = true
    end

    return parse_args(s)
end

# function main()
#     parsed_args = parse_commandline()
#     println("Parsed args:")
#     for pa in parsed_args
#         println("  $(pa[1])  =>  $(pa[2])")
#     end
#     display(parsed_args)
# end

# main() 