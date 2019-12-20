
library(qtl2); 


# get_data <- function(url){
#   data <- read_cross2(url)
#   return(data)
# }
# 
# url = "/Users/xiaoqihu/Documents/hg/lm-gpu/data/spleen/bxd-spleen.yaml"

## read expression traits as character

d <- read.csv(file="/Users/xiaoqihu/Documents/hg/lm-gpu/data/spleen/GN283_MeanDataAnnotated_rev081815.txt",skip=32,
              sep="\t",colClasses="character")

## read genotype file as character
g <- read.csv("/Users/xiaoqihu/Documents/hg/lm-gpu/data/spleen/BXD_current.geno",sep="\t",skip=21,
              colClasses="character")