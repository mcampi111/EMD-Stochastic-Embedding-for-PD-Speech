library(seewave)
library(audio)
library(tuneR)
library(EMD)
library(plyr)
library(zoo)
library(ggplot2)
library(av)
library(foreach)
library(doParallel)
library(snow)
library(reticulate)


dir = "/lustre/home/ucabmc2/workspace/R/EMD_GP/"
setwd(dir)
results_path = paste(dir,'/Results/Parkinson_data/',sep='')
if (!file.exists(results_path)){dir.create(results_path)}

# args = commandArgs(trailingOnly = TRUE)
# print(args)
# print(length(args))
# s = as.numeric(args[1])
# family = as.numeric(args[2])



#IMPORT modules from python
np <- import("numpy")

############################
#     function utils       #        
############################

slice <- function(input, by=2){ 
  starts <- seq(1,length(input),by)
  tt <- lapply(starts, function(y) input[y:(y+(by-1))])
  llply(tt, function(x) x[!is.na(x)])
}

########################################
#           Healthy Patiens            #
########################################

load("/lustre/home/ucabmc2/workspace/R/EMD_GP/Data/hc_sig_batches.RData")

#hc_sig_batches<- lapply(1:hc, function(i) slice(hc_sig[[i]], by = 5000) )

len_batches <- sapply(1:hc, function(i) length(hc_sig_batches[[i]]))


#####################################################
##### extract in parallel the emd decompositions ####
#####################################################


ncores = detectCores() - 1

IMFs_batches_hc<- lapply(1:hc, function(i) vector(mode = "list"))

IMF<- list()


cl = makeCluster(ncores) 
registerDoParallel(cl) 
IMFs_batches_hc<-  foreach(i = 1:1) %:% #hc
  foreach(y = 1:1) %dopar% {  #len_batches[i]
    
    library(EMD)
    IMF<-  emd(as.numeric(hc_sig_batches[[i]][[j]]),
               boundary = "wave", 
               stoprule = "type1",
               max.sift = 20) 
  }

stopCluster(cl)


###################################################
#####  Concate the obtained decomposition  ########
###################################################

imf1<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
imf2<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
imf3<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
lastimf<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
res<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))

for (j in 1: hc) {
  for(i in 1:(length(IMFs_batches_hc[[j]]))){
    
    imf1[[j]]<- c(imf1[[j]], IMFs_batches_hc[[j]][[i]]$imf[,1])
    imf2[[j]]<- c(imf2[[j]], IMFs_batches_hc[[j]][[i]]$imf[,2])
    imf3[[j]]<- c(imf3[[j]], IMFs_batches_hc[[j]][[i]]$imf[,3])
    lastimf[[j]]<- c(lastimf[[j]], IMFs_batches_hc[[j]][[i]]$imf[,(dim(IMFs_batches_hc[[j]][[i]]$imf)[2])])
    res[[j]]<- c(res[[j]], IMFs_batches_hc[[j]][[i]]$residue)
    
  }
}


imf1<- lapply(imf1, function(x) x[!is.na(x)])
imf2<- lapply(imf2, function(x) x[!is.na(x)])
imf3<- lapply(imf3, function(x) x[!is.na(x)])
lastimf<- lapply(lastimf, function(x) x[!is.na(x)])
res<- lapply(res, function(x) x[!is.na(x)])

ft_IMF_hc<- list()

for (j in 1:hc ) {
  ft_IMF_hc[[j]]<- cbind(imf1[[j]], imf2[[j]], imf3[[j]], lastimf[[j]], res[[j]])
}




results_filename = paste(results_path,'/',  'ft_IMF_hc')
saveRDS(ft_IMF_hc, file = results_filename)















