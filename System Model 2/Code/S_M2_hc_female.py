# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:01:32 2021

@author: Marta
"""

##########################
#   Fit System Model 2   #
##########################
#    Healthy Patients    #
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
#mydir_read = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_train_test\\'
#mydir_res = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M2\\Results_Fisher_Kernel\\'


#On Cluster -- modify these
mydir_out = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Logs/HC/'
mydir_read = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Data/'
mydir_res = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Results/HC/'



###################################
#       Functions required        #   
###################################

#def fit_arima_model_parallel(X_train_cent,ind_sld, ind_seg, auto_arima = auto_arima):
#    model = [[auto_arima(X_train_cent[j][ind_sld[i]:ind_sld[i+1]], 
#                    error_action='ignore',
#                    start_p=0, max_p= 3,
#                    start_q= 0, max_q= 3,
#                    d = 1, max_d = 1, m = 1,
#                    seasonal = False) for i in range(0,len(ind_sld)-1)] for j in range(0,ind_seg[1] - ind_seg[0])]
#    return model    



def fit_arima_model_hc_2_parallel(X_train_cent,ind_sld, ind_seg, auto_arima = auto_arima):
    model = [[auto_arima(X_train_cent[j][i], 
                    error_action='ignore',
                    start_p=0, max_p= 3,
                    start_q= 0, max_q= 3,
                    d = 1, max_d = 1, m = 1,
                    max_order = None,
                    seasonal = False,
                    with_intercept = False, trend = None, random = True, n_fits = 2) for i in range(0,len(ind_sld)-1)] for j in range(0,ind_seg[1] - ind_seg[0])]
    return model 


    
##########################
#    Import training     #
##########################

#experiment 1 data:
IMF_train_hc_f = np.load (mydir_read + 'IMF_train_f_hc.npy')
#IMF_train_hc_m = np.load (mydir_read + 'IMF_train_m_hc.npy')


########################
#   Setting Indices    #
########################

N_hc_f = len(IMF_train_hc_f[0])
#N_hc_m = len(IMF_train_hc_m[0])

imf_index = np.int32(sys.argv[2])


#index for sliding window
ind_sld = np.arange(0,5100,100)

#ind_seg = np.concatenate( (np.arange(0, 20, 10), 20), axis = None) 
#ind_seg = np.concatenate( (np.arange(0, N_hc_m, 30), N_hc_m), axis = None)
ind_seg = np.concatenate( (np.arange(0, N_hc_f, 30), N_hc_f), axis = None)

ind1 = np.int32(sys.argv[1])
ind2 = ind1 + 1

ind = [ind_seg[ind1], ind_seg[ind2]]


IMF_train_hc_f_seg = [[ IMF_train_hc_f[imf_index][j][ind_sld[i]:ind_sld[i+1]] for i in range(0, len(ind_sld) -1) ] for j in range(ind[0],ind[1])]



print("The process of centering has started")
#Center the data
tic = time.perf_counter()
IMF_train_hc_f_seg_centr = [ [ (np.array(np.vstack(IMF_train_hc_f_seg[j][i])) - np.mean(IMF_train_hc_f_seg[j][i]))/np.std(IMF_train_hc_f_seg[j][i]) for i in range(0,len(IMF_train_hc_f_seg[j]))]  for j in range(0,len(IMF_train_hc_f_seg))]
#IMF_train_hc_m_seg_centr = [ [ (np.array(np.vstack(IMF_train_hc_m_seg[j][i])) - np.mean(IMF_train_hc_m_seg[j][i]))/np.std(IMF_train_hc_m_seg[j][i]) for i in range(0,len(IMF_train_hc_m_seg[j]))]  for j in range(0,len(IMF_train_hc_m_seg))]
toc = time.perf_counter()
print(f"The process of centering has finished and took {toc - tic:0.4f} seconds")




###########
#On Laptop#
###########

#ind_seg = np.concatenate( (np.arange(0, 50, 10), 50), axis = None) 
#ind_seg = np.concatenate( (np.arange(0, N_hc_m, 100), N_hc_m), axis = None) #Trial!!!!!
#ind1 = ind_seg[0]
#ind2 = ind_seg[1] 
#ind = [ind1, ind2]
#ind

##############
# - normal - #
###############
#res = fit_arima_model_parallel(IMF_train_hc_f_centr[0], ind_sld, ind_seg)
###################
# - in parallel - #
####################
#pool = mp.Pool(2)
#res_2 = pool.apply( fit_arima_model_parallel, args = (IMF_train_hc_m_centr[0], ind_sld, ind) )


################
#On the cluster#
################


if __name__ == '__main__':    #IT WORKS ON THE CLUSTER

     pool = mp.Pool(4)
     tic = time.perf_counter()
     res = pool.apply( fit_arima_model_hc_2_parallel, args = (IMF_train_hc_f_seg_centr, ind_sld, ind) )
     toc = time.perf_counter()
     pool.close()    



print(f"Execute the computation of parallel fit in {toc - tic:0.4f} seconds")



np.save(mydir_res +'res_hc_f_'  + str(ind2) + '_imf_' + str(imf_index) , res)



