# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:37:43 2021

@author: Marta
"""

#################################
#   Cross-Entropy on the IFs    #
#################################
#         IN PARALLEL           #
#################################


import numpy as np
#import multiprocess as mp   # this works only here - then no in the cluster
import multiprocessing as mp # --> use this on the cluster
import time
import sys  
import math
from scipy.stats import gaussian_kde

from RandomPartition import RandomPartition

print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))


##########################
#     Female Patients    #
##########################

########################
#  FUNCTIONS REQUIRED  #
########################

def integrate_box_kde(a,b):
    return kde.integrate_box(a,b)


def evaluate_kde(mesh_positions):
    return kde.evaluate(mesh_positions)


###############################
#      Set Directories        # 
###############################

mydir_out = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Logs/Female_CE/'
mydir_read = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Data/'
mydir_res = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Results/Female_CE/'


#####################
#  Setting Indices  #
#####################

N_f = 354
N_f_test = 89
n_IMF = 3


ind_seg = np.concatenate( (np.arange(0, 10, 5), 10), axis = None)
#ind_seg = np.concatenate( (np.arange(0, N_f, 30), N_f), axis = None)

ind1 = np.int32(sys.argv[1])
ind2 = ind1 + 1

ind = [ind_seg[ind1], ind_seg[ind2]]


# ind1 = ind_seg[0]
# ind2 = ind_seg[1] 
# ind = [ind1, ind2]
# ind


##########################
#       Import Data      #
##########################


#Training Data
IF_train_hc_f_medfit = np.load(mydir_read + "IF_train_hc_f_medfit.npy")
#IF_train_pd_f_medfit = np.load(mydir_read + "IF_train_pd_f_medfit.npy")

#Testing Data
#IF_test_hc_f_medfit = np.load(mydir_read + "IF_test_hc_f_medfit.npy")
#IF_test_pd_f_medfit = np.load(mydir_read + "IF_test_pd_f_medfit.npy")

time_vec = np.load(mydir_read + 'time_vec.npy')


################################################
# CHANGE THIS EVERY TIME OR LOOP OVER THE DATA #
################################################ 

IF = [l[ind[0]:ind[1]] for l in IF_train_hc_f_medfit]

N = len(IF[0])


##################
#GRID PREPARATION#
##################

w = [np.concatenate([IF[0][j],IF[1][j],IF[2][j]]) for j in range(0,N)]
K = 3

data = [ np.c_[np.tile(time_vec*1000,K),w[j] ] for j in range(0,N)]

kde_bdw = 0.1




if __name__ == '__main__':    #IT WORKS ON THE CLUSTER

     tic = time.perf_counter()
     
     partition = []  

     for i in range(0,N):  #chekc 1 sampler 2parallel
      kde = gaussian_kde(data[i].T,bw_method= kde_bdw)
      partition.append(RandomPartition(data[i],3,4,integrate_box_kde,evaluate_kde)  )
      partition[i].estimate_partition_CEM_multinom(n_sim = 50, maxiter_CE = 20,tol_ce = 0.0001, 
                                                  N_grid_omega = 5000, N_grid_t = 1000, tol_omega = 1, tol_t = 1,
                                                  parallel = True, sampler = 'parallel',init_method ='regular')

     
     
     toc = time.perf_counter()



print("Execute the computation of parallel fit in {toc - tic:0.4f} seconds")




np.save(mydir_res +'res_part_f_' + str(ind2) , partition)


