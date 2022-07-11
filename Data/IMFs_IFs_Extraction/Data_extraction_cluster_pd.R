library(EMD)
library(foreach)
library(doParallel)
library(snow)
#library(reticulate)

dir = "/lustre/home/ucabmc2/workspace/R/EMD_GP/"
setwd(dir)
results_path = paste(dir,'/Results/Parkinson_data/',sep='')
if (!file.exists(results_path)){dir.create(results_path)}


#IMPORT modules from python
#np <- import("numpy")


##########################################
#           Parkinson Patiens            #
##########################################

load("/lustre/home/ucabmc2/workspace/R/EMD_GP/Data/pd_sig_batches.RData")

hc = length(pd_sig_batches)

len_batches <- sapply(1:pd, function(i) length(pd_sig_batches[[i]]))


#####################################################
##### extract in parallel the emd decompositions ####
#####################################################


ncores = 30 

IMFs_batches_pd<- lapply(1:hc, function(i) vector(mode = "list"))

IMF<- list()


cl = makeCluster(ncores) 
registerDoParallel(cl) 
IMFs_batches_pd<-  foreach(i = 1:1) %:% #pd
  foreach(y = 1:1) %dopar% {  #len_batches[i]
    
    library(EMD)
    IMF<-  emd(as.numeric(pd_sig_batches[[i]][[j]]),
               boundary = "wave", 
               stoprule = "type1",
               max.sift = 20) 
  }

stopCluster(cl)


###################################################
#####  Concate the obtained decomposition  ########
###################################################

# imf1<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
# imf2<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
# imf3<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
# lastimf<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
# res<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
# 
# for (j in 1: pd) {
#   for(i in 1:(length(IMFs_batches_pd[[j]]))){
#     
#     imf1[[j]]<- c(imf1[[j]], IMFs_batches_pd[[j]][[i]]$imf[,1])
#     imf2[[j]]<- c(imf2[[j]], IMFs_batches_pd[[j]][[i]]$imf[,2])
#     imf3[[j]]<- c(imf3[[j]], IMFs_batches_pd[[j]][[i]]$imf[,3])
#     lastimf[[j]]<- c(lastimf[[j]], IMFs_batches_pd[[j]][[i]]$imf[,(dim(IMFs_batches_pd[[j]][[i]]$imf)[2])])
#     res[[j]]<- c(res[[j]], IMFs_batches_pd[[j]][[i]]$residue)
#     
#   }
# }
# 
# 
# imf1<- lapply(imf1, function(x) x[!is.na(x)])
# imf2<- lapply(imf2, function(x) x[!is.na(x)])
# imf3<- lapply(imf3, function(x) x[!is.na(x)])
# lastimf<- lapply(lastimf, function(x) x[!is.na(x)])
# res<- lapply(res, function(x) x[!is.na(x)])
# 
# ft_IMF_pd<- list()
# 
# for (j in 1:hc ) {
#   ft_IMF_pd[[j]]<- cbind(imf1[[j]], imf2[[j]], imf3[[j]], lastimf[[j]], res[[j]])
# }




results_filename = paste(results_path,'/',  'IMFs_batches_pd') #ft_IMF_pd
saveRDS(IMFs_batches_pd, file = results_filename) #ft_IMF_pd















