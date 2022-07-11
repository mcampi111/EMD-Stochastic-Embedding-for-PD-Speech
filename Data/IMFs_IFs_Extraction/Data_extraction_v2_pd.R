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

##########################################
#           Parkinson Patiens            #
##########################################

#prova<- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results/IMFs_batches_pd_trial.rds")

pd_sig<- np$load("C:/Users/Marta/Desktop/Parkinson_data/Python_code/Data_extraction_&_preliminaries/Results/final_seg_pd.npy",
                allow_pickle = T)

pd = length(pd_sig)


len_pd_sig<- sapply(1:pd, function(i) dim(pd_sig[[i]]))


pd_sig_batches<- lapply(1:pd, function(i) slice(pd_sig[[i]], by = 5000) )

len_batches <- sapply(1:pd, function(i) length(pd_sig_batches[[i]]))

#save(pd_sig_batches, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\pd_sig_batches.RData")
#saveRDS(pd_sig_batches, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\pd_sig_batches.rds")


#########################################################
# remove the batches which relates to the concatenation #
#########################################################


to_remove_pd<- np$load("C:/Users/Marta/Desktop/Parkinson_data/Python_code/Data_extraction_&_preliminaries/Results/to_remove_pd.npy",
                       allow_pickle = T)

pd_sig_batches_final<- lapply(1:pd, function(i) pd_sig_batches[[i]][- c(to_remove_pd[i,])] )

len_batches_final <- sapply(1:pd, function(i) length(pd_sig_batches_final[[i]]))

#save(pd_sig_batches_final, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\pd_sig_batches.RData")

#saveRDS(pd_sig_batches_final,  file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\pd_sig_batches.rds")


############################################
# TOADD here for the version on the cluster
#1 instructions for the cluster
#2 import the file/reading them
#3 at the end save the results
###########################################


#####################################################
##### extract in parallel the emd decompositions ####
#####################################################

# #TRIAL TO MAKE IT WORK
# gne = lapply(1:1, function(i) vector(mode = "list"))
# 
# 
# for(i in 1:1){
#   for(j in 1:1){
# 
#    gne[[i]][[j]] = emd(as.numeric(pd_sig_batches[[i]][[j]]),
#                        boundary = "wave",
#                        stoprule = "type1",
#                        max.sift = 20)
#   }
# }
# 
# 
# 
# imf1<- lapply(1:2, function(x) matrix(NA, nrow=1, ncol=1))
# imf2<- lapply(1:2, function(x) matrix(NA, nrow=1, ncol=1))
# imf3<- lapply(1:2, function(x) matrix(NA, nrow=1, ncol=1))
# lastimf<- lapply(1:2, function(x) matrix(NA, nrow=1, ncol=1))
# res<- lapply(1:2, function(x) matrix(NA, nrow=1, ncol=1))
# 
# 
# 
# for (j in 1: 1) {
#   for(i in 1:(length(gne[[j]]))){
# 
#     imf1[[j]]<- c(imf1[[j]], gne[[j]][[i]]$imf[,1])
#     imf2[[j]]<- c(imf2[[j]], gne[[j]][[i]]$imf[,2])
#     imf3[[j]]<- c(imf3[[j]], gne[[j]][[i]]$imf[,3])
#     lastimf[[j]]<- c(lastimf[[j]], gne[[j]][[i]]$imf[,(dim(gne[[j]][[i]]$imf)[2])])
#     res[[j]]<- c(res[[j]], gne[[j]][[i]]$residue)
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
# for (j in 1:2 ) {
#   ft_IMF_pd[[j]]<- cbind(imf1[[j]], imf2[[j]], imf3[[j]], lastimf[[j]], res[[j]])
# }
# 
# 
# 

#######################
#final for the cluster
######################

ncores = detectCores() - 1

IMFs_batches_pd<- lapply(1:pd, function(i) vector(mode = "list"))

IMF<- list()


cl = makeCluster(ncores) 
registerDoParallel(cl) 
IMFs_batches_pd<-  foreach(i = 1:1) %:% #pd
  foreach(j = 1:1) %dopar% {  #len_batches[i]
    
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


IMFs_batches_pd_1_5 <- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results_final/IMFs_batches_pd_1_5.rds")
IMFs_batches_pd_6_pd <- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results_final/IMFs_batches_pd_6_pd.rds")

IMFs_batches_pd <- do.call(c, list(IMFs_batches_pd_1_5, IMFs_batches_pd_6_pd))

pd = 16


imf1<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
imf2<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
imf3<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
lastimf<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))
res<- lapply(1:pd, function(x) matrix(NA, nrow=1, ncol=1))

for (j in 1: pd) {
  for(i in 1:(length(IMFs_batches_pd[[j]]))){
    
    imf1[[j]]<- c(imf1[[j]], IMFs_batches_pd[[j]][[i]]$imf[,1])
    imf2[[j]]<- c(imf2[[j]], IMFs_batches_pd[[j]][[i]]$imf[,2])
    imf3[[j]]<- c(imf3[[j]], IMFs_batches_pd[[j]][[i]]$imf[,3])
    lastimf[[j]]<- c(lastimf[[j]], IMFs_batches_pd[[j]][[i]]$imf[,(dim(IMFs_batches_pd[[j]][[i]]$imf)[2])])
    res[[j]]<- c(res[[j]], IMFs_batches_pd[[j]][[i]]$residue)
    
  }
}





imf1<- lapply(imf1, function(x) x[!is.na(x)])
imf2<- lapply(imf2, function(x) x[!is.na(x)])
imf3<- lapply(imf3, function(x) x[!is.na(x)])
lastimf<- lapply(lastimf, function(x) x[!is.na(x)])
res<- lapply(res, function(x) x[!is.na(x)])

ft_IMF_pd<- list()

for (j in 1:pd ) {
  ft_IMF_pd[[j]]<- cbind(imf1[[j]], imf2[[j]], imf3[[j]], lastimf[[j]], res[[j]])
}



saveRDS(ft_IMF_pd, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\IMFs_RDS_results_final\\ft_IMF_pd.rds")











