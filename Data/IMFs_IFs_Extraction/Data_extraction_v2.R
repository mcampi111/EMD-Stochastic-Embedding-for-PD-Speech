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

########################################
#           Healthy Patiens            #
########################################

#prova<- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results/IMFs_batches_hc_trial.rds")



hc_sig<- np$load("C:/Users/Marta/Desktop/Parkinson_data/Python_code/Data_extraction_&_preliminaries/Results/final_seg_hc.npy",
                allow_pickle = T)

#load("C:/Users/Marta/Desktop/Parkinson_data/Rcode/hc_sig_batches.RData")

hc = length(hc_sig)


len_hc_sig<- sapply(1:hc, function(i) dim(hc_sig[[i]]))


hc_sig_batches<- lapply(1:hc, function(i) slice(hc_sig[[i]], by = 5000) )

len_batches <- sapply(1:hc, function(i) length(hc_sig_batches[[i]]))

#save(hc_sig_batches, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\hc_sig_batches.RData")

#saveRDS(hc_sig_batches,  file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\hc_sig_batches.rds")


#########################################################
# remove the batches which relates to the concatenation #
#########################################################


to_remove_hc<- np$load("C:/Users/Marta/Desktop/Parkinson_data/Python_code/Data_extraction_&_preliminaries/Results/to_remove_hc.npy",
                 allow_pickle = T)

hc_sig_batches_final<- lapply(1:hc, function(i) hc_sig_batches[[i]][- c(to_remove_hc[i,])] )

len_batches_final <- sapply(1:hc, function(i) length(hc_sig_batches_final[[i]]))


#save(hc_sig_batches_final, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\hc_sig_batches.RData")

#saveRDS(hc_sig_batches_final,  file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\hc_sig_batches.rds")


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
#    gne[[i]][[j]] = emd(as.numeric(hc_sig_batches[[i]][[j]]),
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
# ft_IMF_hc<- list()
# 
# for (j in 1:2 ) {
#   ft_IMF_hc[[j]]<- cbind(imf1[[j]], imf2[[j]], imf3[[j]], lastimf[[j]], res[[j]])
# }
# 
# 
# 

#######################
#final for the cluster
######################

ncores = detectCores() - 1

IMFs_batches_hc<- lapply(1:hc, function(i) vector(mode = "list"))

IMF<- list()


cl = makeCluster(ncores) 
registerDoParallel(cl) 
IMFs_batches_hc<-  foreach(i = 1:hc) %:%
  foreach(j = 1:len_batches[i]) %dopar% {
    
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

IMFs_batches_hc_1_10 <- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results_final/IMFs_batches_hc_1_10.rds")
IMFs_batches_hc_11_hc <- readRDS("C:/Users/Marta/Desktop/Parkinson_data/Rcode/IMFs_RDS_results_final/IMFs_batches_hc_11_hc.rds")

IMFs_batches_hc <- do.call(c, list(IMFs_batches_hc_1_10, IMFs_batches_hc_11_hc))

hc = 21

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


saveRDS(ft_IMF_hc, file = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Rcode\\IMFs_RDS_results_final\\ft_IMF_hc.rds")











