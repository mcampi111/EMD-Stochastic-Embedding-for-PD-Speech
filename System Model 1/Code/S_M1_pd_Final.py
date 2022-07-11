# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:06:16 2021

@author: Marta
"""

##########################
#   Fit System Model 1   #
##########################
#   Parkinson Patients   #
##########################

### Female and Male - do one at a time

import numpy as np
#import multiprocess as mp   # this works only here - then no in the cluster
import multiprocessing as mp # --> use this on the cluster
from pmdarima  import auto_arima
import time
import sys  

print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))



###############################
#      Set Directories        # 
###############################

#On Laptop
#mydir_read = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\New_Training_Testing_Sets\\'
#mydir_res = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M1\\Results\\'


#On Cluster -- modify these
mydir_out = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Logs/PD/Dataset_1/'
mydir_read = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Data/'
mydir_res = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Results/PD/Dataset_1/'



###################################
#       Functions required        #   
###################################

def fit_arima_model_pd_2_parallel(X_train_cent,ind_sld, ind_seg, auto_arima = auto_arima):
    model = [[auto_arima(X_train_cent[j][i], 
                    error_action='ignore',
                    start_p=3, max_p= 3,
                    start_q= 3, max_q= 3,
                    d = 1, max_d = 1, m = 1,
                    max_order = None,
                    seasonal = False,
                    with_intercept = True, trend = 'c', random = True, n_fits = 2) for i in range(0,len(ind_sld)-1)] for j in range(0,ind_seg[1] - ind_seg[0])]
    return model 

    
##########################
#    Import training     #
##########################

X_train_pd_f = np.load (mydir_read + 'X_train_f_pd_d1.npy')
#X_train_pd_m = np.load (mydir_read + 'X_train_m_pd_d1.npy')


########################
#   Setting Indices    #
########################

N_pd_f = len(X_train_pd_f)
#N_pd_m = len(X_train_pd_m)


#index for sliding window
ind_sld = np.arange(0,5100,100)


ind_seg = np.concatenate( (np.arange(0, N_pd_f, 30), N_pd_f), axis = None)
#ind_seg = np.concatenate( (np.arange(0, N_pd_m, 250), N_pd_m), axis = None)

ind1 = np.int32(sys.argv[1])
ind2 = ind1 + 1

ind = [ind_seg[ind1], ind_seg[ind2]]



print("The process of centering has started")

#Center the data
tic = time.perf_counter()
X_train_pd_f_centr = [ (np.array(np.vstack(X_train_pd_f[i])) - np.mean(X_train_pd_f[i]))/np.std(X_train_pd_f[i]) for i in range(ind[0],ind[1])] # 50 250
#X_train_pd_m_centr = [ (np.array(np.vstack(X_train_pd_m[i])) - np.mean(X_train_pd_m[i]))/np.std(X_train_pd_m[i])  for i in range(ind[0],ind[1])] # N_pd_m
toc = time.perf_counter()
print("The process of centering has finished and took {toc - tic:0.4f} seconds")


###########
#On Laptop#
###########

#ind_seg = np.concatenate( (np.arange(0, 50, 10), 50), axis = None) 
#ind_seg = np.concatenate( (np.arange(0, N_pd_m, 100), N_pd_m), axis = None) #Trial!!!!!
#ind1 = ind_seg[0]
#ind2 = ind_seg[1] 
#ind = [ind1, ind2]
#ind

##############
# - normal - #
###############
#res = fit_arima_model_parallel(X_train_pd_m_centr, ind_sld, ind_seg)
###################
# - in parallel - #
####################
#pool = mp.Pool(2)
#res_2 = pool.apply( fit_arima_model_parallel, args = (X_train_pd_m_centr, ind_sld, ind) )



################
#On the cluster#
################


if __name__ == '__main__':    #IT WORKS ON THE CLUSTER

     pool = mp.Pool(4)
     tic = time.perf_counter()
     res = pool.apply( fit_arima_model_pd_2, args = (X_train_pd_f_centr, ind_sld, ind) )
     toc = time.perf_counter()
     pool.close()    



print(f"Execute the computation of parallel fit in {toc - tic:0.4f} seconds")



np.save(mydir_res +'res_pd_f_' + 'exp1_'  + str(ind2) , res)


