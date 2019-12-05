library(mice)
# library(lattice)
library(parallel)
library(qtl2)

getdata<-function(url){
  return bxd <- read.cross2(url)
}

keep_row<-function(data, droprate){
  rs = rowSums(is.na(data$pheno)) 
  keepidx <- which(rs < quantile(rs, 1-droprate/100))
  return subset(data,ind=keepidx)
}

drop_col<-function(data, droprate){
  rownames(data$pheno)<-data$pheno$ID
  data$pheno<-data$pheno[,-1]

  trait = data$pheno
  end<-dim(trait)[2]

  cs = colSums(is.na(trait[,2:(end-1)]))
  drop.idx<-which(cs > quantile(cs, 1-droprate/100)
  trait<-trait[,2:(end-1)]
  trait<-trait[,-drop.idx]
  return trait
}

calc_geno_prob<-function(sub_cross, ncore, error_prob, step, pseudomarker){

  #insert pseudomarker
  map <- insert_pseudomarkers(sub_cross$gmap, step=step)
  pr <- calc_genoprob(sub_cross, map, error_prob=error_prob, cores=ncores)
}

#get whole genotype prob file
getGenopr<-function(x){
  temp<<-NULL
  m=length(attributes(x)$names)
  cnames<-attributes(x)$names
  for (i in 1:m) {
    d<-eval(parse(text=paste(c('dim(x$\'', cnames[i] ,'\')'),collapse='')))
    nam<-eval(parse(text=paste(c('dimnames(x$\'',cnames[i],'\')[[2]]'),collapse = '')))
    cnam<-rep(nam,d[3])
    p_chr<-paste(c('array(x$\'',cnames[i],'\',dim=c(d[1],d[2]*d[3]))'),collapse='')
    prob<-eval(parse(text = p_chr))
    temp<-cbind(temp,prob)
  }
  return(temp)
}

# url : data url
# indi_droprate: droprate in percentage, ie: 10 percent
# trait_droprate : droprate in percentage, ie: 10 percent
# ncores: default detectCores()
cleaning<-function(url, indi_droprate, trait_droprate, nseed, ncores, error_prob, step){  
  bxd = getdata(url)

  # process pheno
  sub_cross = keep_row(bxd, indi_droprate)
  trait = keep_col(sub_cross, trait_droprate)

  #imputation
  nseed = 10
  temp_imp = mice(trait, defaultMethod = "pmm", seed = nseed)
  imp = complete(temp_imp)

  # calculate genotype probablity
  ncores = detectCores()
  error_prob = 0.002
  step = 1
  pr = calc_geno_prob(sub_cross, ncores, error_prob, step)
  prob1 = getGenopr(pr)

}

