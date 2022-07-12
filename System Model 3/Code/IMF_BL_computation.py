# -*- coding: utf-8 -*-
"""
@author: Marta
"""

import numpy as np
import os as os
import time
import seaborn as sn
from matplotlib import cm as CM
import math
from scipy.stats import gaussian_kde
import natsort                                                                                              
from pathlib import Path
import pandas as pd


os.getcwd()
os.chdir('C:/Users/Marta/Desktop/Parkinson_data/Python_code/System_M3/Code')

from RandomPartition import RandomPartition


#EXTRACTION BAND-LIMITED IMFs


#######################
# FUNCTIONS REQUIRED  #
#######################

def integrate_box_kde(a,b):
    return kde.integrate_box(a,b)


def evaluate_kde(mesh_positions):
    return kde.evaluate(mesh_positions)



def IMF_BL_extraction(vec_partition, IFs_list, IMFs_list):  #you migh have to add the lopp for the segments
    IMFs_sep = [[np.zeros(5000), np.zeros(5000), np.zeros(5000)] for i in range(0,3)]
    ind_imfs = [ [np.where((vec_partition[0] <= IFs_list[j]) &  (IFs_list[j] < vec_partition[1]))[0],
                  np.where((vec_partition[1] < IFs_list[j]) &  (IFs_list[j] < vec_partition[2]))[0], 
                  np.where((vec_partition[2] < IFs_list[j]) &  (IFs_list[j] <= vec_partition[3]))[0] ] for j in range(0,3)] 
    for j in range(0,3):
        for i in range(0,3):
            IMFs_sep[j][i][ind_imfs[j][i]] = IMFs_list[j][ind_imfs[j][i]]
            
    
    IMF_BL = [sum(x) for x in zip(*IMFs_sep)] 
    return(IMF_BL)   



###############################
#     Reading CE Results      # 
###############################

os.getcwd()
ce_files = [x for x in [_ for _ in Path('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M3\\Results_CE\\Female\\HC\\').rglob('*.npy') if _.is_file()] ]
ce_files_ordered =  natsort.natsorted(ce_files, key=str)  


ce_res = [np.load(ce_files_ordered[i], allow_pickle=True) for i in  range(0,len(ce_files_ordered))] 

[ len(ce_res[i]) for i in  range(0,len(ce_files_ordered))] 


#PD FOLDER
#use this once other patients are extracted for testing
ce_res_train = [ce_res[i] for i in range(0,12)]
ce_res_test =  [ce_res[i] for i in range(12,15)] # this is patient 1 status 1

ce_res_train = [item for sublist in ce_res_train for item in sublist]
ce_res_test = [item for sublist in ce_res_test for item in sublist]

#HC FOLDER
#use this once other patients are extracted for testing

ce_res_train= ce_res[1]
ce_res_test= ce_res[0]  # this is patient 0 status 0


###############################
#      Set Directories        # 
###############################

mydir_read_imf = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_train_test\\'
mydir_read_if = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IFs_train_test\\'

####################################
#      Loading IFs and IMFs        # 
####################################

IF_train_hc_f_med = np.load(mydir_read_if + "IF_train_hc_f_medfit.npy",
                            allow_pickle = True)

IMF_train_f_hc =  np.load(mydir_read_imf + "IMF_train_f_hc.npy", 
                            allow_pickle = True)

IF_train_pd_f_med = np.load(mydir_read_if + "IF_train_pd_f_medfit.npy",
                            allow_pickle = True)

IMF_train_f_pd =  np.load(mydir_read_imf + "IMF_train_f_pd_d1.npy", 
                            allow_pickle = True)

#this depends on the testing patient
IF_test_hc_f_med = np.load(mydir_read_if + "IF_test_hc_f.npy", #this is pat 0 status 0
                            allow_pickle = True)

#this depends on the testing patient
IMF_test_f_hc =  np.load(mydir_read_imf + "IMF_test_hc_f.npy", #this is pat 0 status 0
                            allow_pickle = True)

#this depends on the testing patient
IF_test_pd_f_med = np.load(mydir_read_if + "IF_test_pd_f_medfit.npy", #this is pat 1 status 1
                            allow_pickle = True)

#this depends on the testing patient
IMF_test_f_pd =  np.load(mydir_read_imf + "IMF_test_pd_f_d1.npy", #this is pat 1 status 1
                            allow_pickle = True)


########################
#   Extracting IMF-BL  #
########################

N_seg = len(ce_res_train)
N_seg_test = len(ce_res_test)


IF_f_pd_train = [ [IF_train_pd_f_med[0][i], IF_train_pd_f_med[1][i],IF_train_pd_f_med[2][i]] for i in range(0,N_seg)]
IMF_f_pd_train = [ [IMF_train_f_pd[0][i], IMF_train_f_pd[1][i],IMF_train_f_pd[2][i]] for i in range(0,N_seg)]


breaks_train = [  ce_res_train[i].breaks_old[0] for i in range(0,N_seg)]

IMF_BL_pd_f_train = [IMF_BL_extraction(breaks_train[i], IF_f_pd_train[i], IMF_f_pd_train[i]) for i in range(0,N_seg)]

np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_pd_f_train.npy', 
        IMF_BL_pd_f_train)




IF_f_hc_train = [ [IF_train_hc_f_med[0][i], IF_train_hc_f_med[1][i],IF_train_hc_f_med[2][i]] for i in range(0,N_seg)]
IMF_f_hc_train = [ [IMF_train_f_hc[0][i], IMF_train_f_hc[1][i],IMF_train_f_hc[2][i]] for i in range(0,N_seg)]


breaks_train = [  ce_res_train[i].breaks_old[0] for i in range(0,N_seg)]

IMF_BL_hc_f_train = [IMF_BL_extraction(breaks_train[i], IF_f_hc_train[i], IMF_f_hc_train[i]) for i in range(0,N_seg)]

np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_hc_f_train.npy', 
        IMF_BL_hc_f_train)




#this depends on the testing patient
#patient 0 status 0
IF_f_hc_test = [ [IF_test_hc_f_med[0][i], IF_test_hc_f_med[1][i],IF_test_hc_f_med[2][i]] for i in range(0,N_seg_test)]
IMF_f_hc_test = [ [IMF_test_f_hc[0][i], IMF_test_f_hc[1][i],IMF_test_f_hc[2][i]] for i in range(0,N_seg_test)]


breaks_test = [  ce_res_test[i].breaks_old[0] for i in range(0,N_seg_test)]

IMF_BL_hc_f_test = [IMF_BL_extraction(breaks_test[i], IF_f_hc_test[i], IMF_f_hc_test[i]) for i in range(0,N_seg_test)]


np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_hc_f_test.npy', 
        IMF_BL_hc_f_test)



#this depends on the testing patient
#patient 1 status 1
IF_f_pd_test = [ [IF_test_pd_f_med[0][i], IF_test_pd_f_med[1][i],IF_test_pd_f_med[2][i]] for i in range(0,N_seg_test)]
IMF_f_pd_test = [ [IMF_test_f_pd[0][i], IMF_test_f_pd[1][i],IMF_test_f_pd[2][i]] for i in range(0,N_seg_test)]


breaks_test = [  ce_res_test[i].breaks_old[0] for i in range(0,N_seg_test)]

IMF_BL_pd_f_test = [IMF_BL_extraction(breaks_test[i], IF_f_pd_test[i], IMF_f_pd_test[i]) for i in range(0,N_seg_test)]


np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_pd_f_test.npy', 
        IMF_BL_pd_f_test)






##################
#      MALE      #
##################


###############################
#     Reading CE Results      # 
###############################

#change PD or HC
os.getcwd()
ce_files = [x for x in [_ for _ in Path('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M3\\Results_CE\\Male\\HC\\').rglob('*.npy') if _.is_file()] ]
ce_files_ordered =  natsort.natsorted(ce_files, key=str)  


ce_res = [np.load(ce_files_ordered[i], allow_pickle=True) for i in  range(0,len(ce_files_ordered))] 

[ len(ce_res[i]) for i in  range(0,len(ce_files_ordered))] 


#PD FOLDER
#use this once other patients are extracted for testing
ce_res_train = [ce_res[i] for i in range(0,10)]
ce_res_test =  [ce_res[i] for i in range(10,13)] # this is patient 1 status 1

ce_res_train = [item for sublist in ce_res_train for item in sublist]
ce_res_test = [item for sublist in ce_res_test for item in sublist]

#HC FOLDER
#use this once other patients are extracted for testing

ce_res_train = [ce_res[i] for i in range(0,10)]
ce_res_test =  [ce_res[i] for i in range(10,13)] # this is patient 0 status 0

ce_res_train = [item for sublist in ce_res_train for item in sublist]
ce_res_test = [item for sublist in ce_res_test for item in sublist]

###############################
#      Set Directories        # 
###############################

mydir_read_imf = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_train_test\\'
mydir_read_if = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IFs_train_test\\'

####################################
#      Loading IFs and IMFs        # 
####################################

IF_train_hc_m_med = np.load(mydir_read_if + "IF_train_hc_m_medfit.npy",
                            allow_pickle = True)

IMF_train_m_hc =  np.load(mydir_read_imf + "IMF_train_m_hc.npy", 
                            allow_pickle = True)

IF_train_pd_m_med = np.load(mydir_read_if + "IF_train_pd_m_medfit.npy",
                            allow_pickle = True)

IMF_train_m_pd =  np.load(mydir_read_imf + "IMF_train_m_pd_d1.npy", 
                            allow_pickle = True)

#this depends on the testing patient
IF_test_hc_m_med = np.load(mydir_read_if + "IF_test_hc_m.npy", #this is pat 0 status 0
                            allow_pickle = True)

#this depends on the testing patient
IMF_test_m_hc =  np.load(mydir_read_imf + "IMF_test_hc_m.npy", #this is pat 0 status 0
                            allow_pickle = True)

#this depends on the testing patient
IF_test_pd_m_med = np.load(mydir_read_if + "IF_test_pd_m_medfit.npy", #this is pat 1 status 1
                            allow_pickle = True)

#this depends on the testing patient
IMF_test_m_pd =  np.load(mydir_read_imf + "IMF_test_pd_m_d1.npy", #this is pat 1 status 1
                            allow_pickle = True)


########################
#   Extracting IMF-BL  #
########################

N_seg = len(ce_res_train)
N_seg_test = len(ce_res_test)


IF_m_pd_train = [ [IF_train_pd_m_med[0][i], IF_train_pd_m_med[1][i],IF_train_pd_m_med[2][i]] for i in range(0,N_seg)]
IMF_m_pd_train = [ [IMF_train_m_pd[0][i], IMF_train_m_pd[1][i],IMF_train_m_pd[2][i]] for i in range(0,N_seg)]


breaks_train = [  ce_res_train[i].breaks_old[0] for i in range(0,N_seg)]

IMF_BL_pd_m_train = [IMF_BL_extraction(breaks_train[i], IF_m_pd_train[i], IMF_m_pd_train[i]) for i in range(0,N_seg)]

np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_pd_m_train.npy', 
        IMF_BL_pd_m_train)




IF_m_hc_train = [ [IF_train_hc_m_med[0][i], IF_train_hc_m_med[1][i],IF_train_hc_m_med[2][i]] for i in range(0,N_seg)]
IMF_m_hc_train = [ [IMF_train_m_hc[0][i], IMF_train_m_hc[1][i],IMF_train_m_hc[2][i]] for i in range(0,N_seg)]


breaks_train = [  ce_res_train[i].breaks_old[0] for i in range(0,N_seg)]

IMF_BL_hc_m_train = [IMF_BL_extraction(breaks_train[i], IF_m_hc_train[i], IMF_m_hc_train[i]) for i in range(0,N_seg)]

np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_hc_m_train.npy', 
        IMF_BL_hc_m_train)




#this depends on the testing patient
#patient 0 status 0
IF_m_hc_test = [ [IF_test_hc_m_med[0][i], IF_test_hc_m_med[1][i],IF_test_hc_m_med[2][i]] for i in range(0,N_seg_test)]
IMF_m_hc_test = [ [IMF_test_m_hc[0][i], IMF_test_m_hc[1][i],IMF_test_m_hc[2][i]] for i in range(0,N_seg_test)]


breaks_test = [  ce_res_test[i].breaks_old[0] for i in range(0,N_seg_test)]


IMF_BL_hc_m_test = [IMF_BL_extraction(breaks_test[i], IF_m_hc_test[i], IMF_m_hc_test[i]) for i in range(0,N_seg_test)]


np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_hc_m_test.npy', 
        IMF_BL_hc_m_test)



#this depends on the testing patient
#patient 1 status 1
IF_m_pd_test = [ [IF_test_pd_m_med[0][i], IF_test_pd_m_med[1][i],IF_test_pd_m_med[2][i]] for i in range(0,N_seg_test)]
IMF_m_pd_test = [ [IMF_test_m_pd[0][i], IMF_test_m_pd[1][i],IMF_test_m_pd[2][i]] for i in range(0,N_seg_test)]


breaks_test = [  ce_res_test[i].breaks_old[0] for i in range(0,N_seg_test)]

IMF_BL_pd_m_test = [IMF_BL_extraction(breaks_test[i], IF_m_pd_test[i], IMF_m_pd_test[i]) for i in range(0,N_seg_test)]


np.save('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\IMF_BL_train_test\\IMF_BL_pd_m_test.npy', 
        IMF_BL_pd_m_test)





