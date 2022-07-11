library(seewave)
library(audio)
library(tuneR)
library(EMD)
library(foreach)
library(doParallel)
library(snow)
library(reticulate)
library(gtools)

############################
#IMPORT modules from python#
############################

np <- import("numpy")

########################
# SETTING DIRECTORIES  #
########################

# dir = "/lustre/home/ucabmc2/workspace/R/EMD_GP/"
# setwd(dir)
# results_path = paste(dir,'/Results/Parkinson_data/',sep='')
# if (!file.exists(results_path)){dir.create(results_path)}
#load_path = "/lustre/home/ucabmc2/workspace/R/EMD_GP/Data/"

#ON LAPTOP
load_path = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_train_test\\"
load_path2 = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_train_test\\Test_last_extracted\\"
res_path = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IFs_train_test\\"
res_path2 = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IFs_train_test\\Test_last_extracted\\"


########################
#  SETTING ARGUMENTS   #
########################
# args = commandArgs(trailingOnly = TRUE)
# print(args)
# print(length(args))
# index_imf =  as.numeric(args[1])
# index_seg = as.numeric(args[2])

fs = 44100
n_IMF = 5

N_f = 354
N_m = 298

#################
#  IMPORT IMFs  #
#################

#TRAINING

IMF_train_hc_f<- np$load(paste( load_path, 'IMF_train_f_hc.npy', sep = "")) #IMF_test_hc_f.npy
IMF_train_hc_m<- np$load(paste( load_path, 'IMF_train_m_hc.npy', sep = "")) #IMF_test_hc_m.npy
IMF_train_pd_f<- np$load(paste( load_path, 'IMF_train_f_pd_d1.npy', sep = "")) #IMF_test_pd_f.npy
IMF_train_pd_m<- np$load(paste( load_path, 'IMF_train_m_pd_d1.npy', sep = "")) #IMF_test_pd_m.npy



IF_train_hc_f<- lapply(1:n_IMF,function(i)lapply(1:N_f, function(j) 
                   ifreq( wave = IMF_train_hc_f[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )

IF_train_hc_m<- lapply(1:n_IMF,function(i)lapply(1:N_m, function(j) 
          ifreq( wave = IMF_train_hc_m[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )


IF_train_pd_f<- lapply(1:n_IMF,function(i)lapply(1:N_f, function(j) 
               ifreq( wave = IMF_train_pd_f[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )

IF_train_pd_m<- lapply(1:n_IMF,function(i)lapply(1:N_m, function(j) 
                ifreq( wave = IMF_train_pd_m[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )



prova = do.call("cbind",IF_train_hc_f)


IF_train_hc_f_final<- lapply(1:n_IMF,function(i)lapply(1:N_f, function(j) IF_train_hc_f[[i]][[j]]$f))
IF_train_hc_m_final<- lapply(1:n_IMF,function(i)lapply(1:N_m, function(j) IF_train_hc_m[[i]][[j]]$f))
IF_train_pd_f_final<- lapply(1:n_IMF,function(i)lapply(1:N_f, function(j) IF_train_pd_f[[i]][[j]]$f))
IF_train_pd_m_final<- lapply(1:n_IMF,function(i)lapply(1:N_m, function(j) IF_train_pd_m[[i]][[j]]$f))


#how to access them on python:
#np.array(IF_train_hc_f[0][0])[:,1]

np$save(paste(res_path, 'IF_train_hc_f', sep = ''), IF_train_hc_f_final)
np$save(paste(res_path, 'IF_train_hc_m', sep = ''), IF_train_hc_m_final)
np$save(paste(res_path, 'IF_train_pd_f', sep = ''), IF_train_pd_f_final)
np$save(paste(res_path, 'IF_train_pd_m', sep = ''), IF_train_pd_m_final)



#TESTING

N_f_test = 89
N_m_test = 75


IMF_test_hc_f<- np$load(paste( load_path, 'IMF_test_hc_f.npy', sep = "")) 
IMF_test_hc_m<- np$load(paste( load_path, 'IMF_test_hc_m.npy', sep = "")) 
IMF_test_pd_f<- np$load(paste( load_path, 'IMF_test_pd_f_d1.npy', sep = "")) 
IMF_test_pd_m<- np$load(paste( load_path, 'IMF_test_pd_m_d1.npy', sep = "")) 



IF_test_hc_f<- lapply(1:n_IMF,function(i)lapply(1:N_f_test, function(j) 
  ifreq( wave = IMF_test_hc_f[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )

IF_test_hc_m<- lapply(1:n_IMF,function(i)lapply(1:N_m_test, function(j) 
  ifreq( wave = IMF_test_hc_m[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )


IF_test_pd_f<- lapply(1:n_IMF,function(i)lapply(1:N_f_test, function(j) 
  ifreq( wave = IMF_test_pd_f[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )

IF_test_pd_m<- lapply(1:n_IMF,function(i)lapply(1:N_m_test, function(j) 
  ifreq( wave = IMF_test_pd_m[i,j,], f = fs, plot = FALSE, phase = FALSE) ) )



IF_test_hc_f_final<- lapply(1:n_IMF,function(i)lapply(1:N_f_test, function(j) IF_test_hc_f[[i]][[j]]$f))
IF_test_hc_m_final<- lapply(1:n_IMF,function(i)lapply(1:N_m_test, function(j) IF_test_hc_m[[i]][[j]]$f))
IF_test_pd_f_final<- lapply(1:n_IMF,function(i)lapply(1:N_f_test, function(j) IF_test_pd_f[[i]][[j]]$f))
IF_test_pd_m_final<- lapply(1:n_IMF,function(i)lapply(1:N_m_test, function(j) IF_test_pd_m[[i]][[j]]$f))


#how to access them on python:
#np.array(IF_test_hc_f[0][0])[:,1]

np$save(paste(res_path, 'IF_test_hc_f', sep = ''), IF_test_hc_f_final)
np$save(paste(res_path, 'IF_test_hc_m', sep = ''), IF_test_hc_m_final)
np$save(paste(res_path, 'IF_test_pd_f', sep = ''), IF_test_pd_f_final)
np$save(paste(res_path, 'IF_test_pd_m', sep = ''), IF_test_pd_m_final)


###########################
# NEW TEST IMFS EXTRACTED #
###########################


my_files = mixedsort(list.files(load_path2))

status = c('hc','pd')
gender = c('f', 'm')


status_for_names = c('hc', 'hc', 'pd', 'pd')
gender_for_names = c('f', 'm', 'f', 'm')


hc_f_pat = c(0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18)
hc_m_pat = 0

pd_f_pat = c(0,3)
pd_m_pat =c(0,1,2,3,4,6,7,9,10,11)


all_patients = list(hc_f_pat, hc_m_pat, pd_f_pat,pd_m_pat)

IF_names= lapply(1:length(all_patients), function(i) 
              sapply(1:length(all_patients[[i]]), function(j)
                 paste("IF_test", status_for_names[i],  gender_for_names[i], all_patients[[i]][j], sep = "_"  )))



new_length_myfile = sapply(all_patients, length)


my_files_new = split(my_files, rep(1:4, new_length_myfile))

IMFs_test = lapply(1:length(my_files_new), function(i)
                 lapply(1:length(my_files_new[[i]]), function(j)
                   np$load( paste(load_path2,my_files_new[[i]][[j]],sep = '' ) )))



IFs_test<- lapply(1:length(my_files_new), function(i)
                lapply(1:length(my_files_new[[i]]), function(j) 
                  lapply(1:n_IMF,function(h)lapply(1:(dim(IMFs_test[[i]][[j]])[2]), function(k)
                      ifreq( wave = IMFs_test[[i]][[j]][h,k,], f = fs, plot = FALSE, phase = FALSE)  ))))
  
  


IFs_test[[1]][[1]][[1]][[1]] #indices gives 1) type of lists 2) patient 3) IMFs 4) Segments


IFs_test_2<- lapply(1:length(my_files_new), function(i)
               lapply(1:length(my_files_new[[i]]), function(j) 
                 lapply(1:n_IMF,function(h)lapply(1:(dim(IMFs_test[[i]][[j]])[2]), function(k)
                                                  IFs_test[[i]][[j]][[h]][[k]]$f ))))



for (i in 1:length(my_files_new)) {
  for (j in 1:length(my_files_new[[i]])) {
    
    np$save(paste(res_path2, IF_names[[i]][[j]], sep = ''), IFs_test_2[[i]][[j]])
    
  }
  
}





