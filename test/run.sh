#!/bin/bash

geno_file="/Users/xiaoqihu/Documents/hg/genome-scan-data-cleaning/test/geno_prob.csv"

pheno_file="/Users/xiaoqihu/Documents/hg/genome-scan-data-cleaning/test/imputed_pheno.csv"

julia test.jl --geno_file=$geno_file --pheno_file=$pheno_file
