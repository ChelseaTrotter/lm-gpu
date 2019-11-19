using RCall 

# RIL by selfing 

# read yaml file
# library(qtl2)
# grav2 <- read_cross2("https://kbroman.org/qtl2/assets/sampledata/grav2/grav2.yaml")
# R"""
# library(vioplot); agrenViolinPlot <- function(){
# agrenURL <- "https://bitbucket.org/linen/smalldata/raw/3c9bcd603b67a16d02c5dc23e7e3e637758d4d5f/arabidopsis/agren2013.csv"
# agren <- read.csv(agrenURL); agrenFit <- agren[,c(1,2,3,4,5,6)]
# vioplot(agrenFit, names=names(agrenFit), main = "ViolinPlot of Fitness Values per site and year", xlab ="Site", ylab =  "Fitness Values",col = rainbow(6))}
# """
# R"agrenViolinPlot"();

# ARGS is the Julia Macro for commandline arguments.
url = ARGS

R"""
library(qtl2); get_data <- function(){
data <- read_cross2($url)
return(data)
}
"""
data = R"get_data"()


# read in data with zip file by url
# library(qtl2)
# grav2 <- read_cross2("https://kbroman.org/qtl2/assets/sampledata/grav2/grav2.zip")
# zip the data

display(data)
names(data)
geno = data[:geno];
pheno = data[:pheno];

