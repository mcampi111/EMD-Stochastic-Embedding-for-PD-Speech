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

############################################################
####### Speech-to-Text - need to check this ################
############################################################
#library(warbleR)
#library(transcribeR)
#library(httr)
#library(googleLanguageR)
#library(reticulate)
#Use the following when R cannot find the Module of interest
# use_condaenv("r-reticulate")
# py_install("SpeechRecognition", pip = TRUE)
# py_module_available("speech_recognition")
# sr<- import("speech_recognition")
# os<- import("os")
# pydub<- import("pydub")
#source_python("C:\\Users\\Marta\\Desktop\\Parkinson_data\\26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\test_func.py")
############################################################

########################################
#how to read numpy list of arrays in R #
########################################
library(reticulate)
np <- import("numpy")
prova<- np$load("C:/Users/Marta/Desktop/Parkinson_data/Python_code/Data_extraction_&_preliminaries/Results/final_seg_hc.npy",
                allow_pickle = T)
prova2<- lapply(1: length(prova), function(i) data.frame(prova[i]))

########################################
#           Healthy Patiens            #
########################################

setwd("C:\\Users\\Marta\\Desktop\\Parkinson_data\\26_29_09_2017_KCL\\ReadText\\HC")

mydir<- list.files(getwd())

hc = length(mydir)

#####################################################
#Instructions:
#we extract emd in batches of 20 seconds for the entire signal
#REMARK: We will align sentences in a second moment
#CHECK WITH G/D IF read_audio_bin works ok

#CHECK THE RAM THAT I M REQUESTING

durations = sapply(1:21, function(i) av_media_info(mydir[[i]])$duration)

s_freq = 44100

n_samples = durations*s_freq

ty_sec = 883155 # read_audio_bin(mydir[[1]], start_time = 0, end_time = 20)

lecture_hc <- lapply(1:hc, function(i) read_audio_bin(mydir[[i]]))


lapply(seq_along(com.list), function(i) {
  dat <- com.list[[i]]
  subset(dat, row.names(dat) %in% years[,i])
})

#Produce a list of matrices which have in columns 20 secs frames for the emd (20 secs 
#correspond to 883155 number of samples)
data_hc_frames <- lapply(1:hc, function(j){ 
                         sapply( c(seq(0, n_samples[j], by = ty_sec)), function(i){
                         lecture_hc[[j]][(i+1):(i+ty_sec)]
  })
})




#####################################################
##### extract in parallel the emd decompositions ####
#####################################################

#test with lapply
# IMFs_frame_hc<- lapply(1:hc, function(x) lapply( 1:(dim(data_hc_frames[[x]])[2]), function(y) 
#                                                   emd(as.numeric(data_hc_frames[[x]][1:10,y]),
#                                                        boundary = "wave", 
#                                                        stoprule = "type1",
#                                                        max.sift = 20)  ) )
# 

#test with only the first 10 observations
# cl = makeCluster(ncores) 
# registerDoParallel(cl) 
# IMFs_frame_hc<-  foreach(x = 1:hc) %:%
#   foreach(y = 1:(dim(data_hc_frames[[x]])[2])) %dopar% {
#     
#     library(EMD)
#     IMF<- emd(as.numeric(data_hc_frames[[x]][1:10,y]),
#               boundary = "wave", 
#               stoprule = "type1",
#               max.sift = 20)  
#   }
# 
# stopCluster(cl)


ncores = detectCores() - 1


#Need to initialise 
IMFs_frame_hc<- lapply(1:hc, function(x) lapply( 1:(dim(data_hc_frames[[x]])[2]), 
                                                 function(y) vector(mode = "list")  ) )

IMF<- list()

cl = makeCluster(ncores) 
registerDoParallel(cl) 
IMFs_frame_hc<-  foreach(x = 1:hc) %:%
  foreach(y = 1:(dim(data_hc_frames[[x]])[2])) %dopar% {
    
    library(EMD)
    IMF<- emd(as.numeric(data_hc_frames[[x]][,y]),
              boundary = "wave", 
              stoprule = "type1",
              max.sift = 20)  
  }

stopCluster(cl)

###################################################
#####  Concate the obtained decomposition  ########
###################################################


##########################################################
#check if the loop breaks since there are not enough IMFs#
##########################################################


imf1<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
imf2<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
imf3<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
lastimf<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))
res<- lapply(1:hc, function(x) matrix(NA, nrow=1, ncol=1))

for (j in 1: hc) {
  for(i in 1:(length(IMFs_frame_hc[[j]]))){
    
    imf1[[j]]<- c(imf1[[j]], IMFs_frame_hc[[j]][[i]]$imf[,1])
    imf2[[j]]<- c(imf2[[j]], IMFs_frame_hc[[j]][[i]]$imf[,2])
    imf3[[j]]<- c(imf3[[j]], IMFs_frame_hc[[j]][[i]]$imf[,3])
    lastimf[[j]]<- c(lastimf[[j]], IMFs_frame_hc[[j]][[i]]$imf[,(dim(IMFs_frame_hc[[j]][[i]]$imf)[2])])
    res[[j]]<- c(res[[j]], IMFs_frame_hc[[j]][[i]]$residue)
    
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





