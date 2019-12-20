using DelimitedFiles
function extract_gmap()
    # geno_file = readdlm("../data/spleen/BXD_current.geno", ' '; skipstart=21)[:,2:end-1]
    gmap = readdlm("../data/spleen/BXD_current.geno", '\t'; skipstart=21)[:,1:4]
    idx = ["id"; collect(1:1:size(gmap,1)-1)]
    writedlm("../data/spleen/gmap.csv", [idx gmap], ',')
end

function process_traits()
    traits = readdlm("../data/spleen/GN283_MeanDataAnnotated_rev081815.txt", '\t'; comments=true, comment_char='#')
    
end

# gmap = extract_gmap()
traits = process_traits()