# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:05:14 2021

@author: Marta
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:01:46 2021

@author: Marta
"""

import numpy as np
import os as os
import natsort                                                                                              
from pathlib import Path
import matplotlib.pyplot as plt
import time
import seaborn as sn
from matplotlib import cm as CM
import math
from pmdarima  import auto_arima
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import math
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import norm

from rpy2.robjects import r
from scipy.stats.distributions import chi2
from scipy.stats import chi2


from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from statsmodels.tsa.innovations import _arma_innovations



#######################
# FUNCTIONS REQUIRED  #
#######################

NON_STATIONARY_ERROR = """\
The model's autoregressive parameters (ar_params) indicate that the process
 is non-stationary. The innovations algorithm cannot be used.
"""


def arma_innovations(endog, ar_params=None, ma_params=None, sigma2=1,
                     normalize=False, prefix=None):
    """
    Compute innovations using a given ARMA process.
    Parameters
    ----------
    endog : ndarray
        The observed time-series process, may be univariate or multivariate.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    normalize : bool, optional
        Whether or not to normalize the returned innovations. Default is False.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.
    Returns
    -------
    innovations : ndarray
        Innovations (one-step-ahead prediction errors) for the given `endog`
        series with predictions based on the given ARMA process. If
        `normalize=True`, then the returned innovations have been "whitened" by
        dividing through by the square root of the mean square error.
    innovations_mse : ndarray
        Mean square error for the innovations.
    """
    # Parameters
    endog = np.array(endog)
    squeezed = endog.ndim == 1
    if squeezed:
        endog = endog[:, None]

    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    nobs, k_endog = endog.shape
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # Get BLAS prefix
    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]

    # Make arrays contiguous for BLAS calls
    endog = np.asfortranarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = dtype(sigma2).item()

    # Get the appropriate functions
    arma_transformed_acovf_fast = getattr(
        _arma_innovations, prefix + 'arma_transformed_acovf_fast')
    arma_innovations_algo_fast = getattr(
        _arma_innovations, prefix + 'arma_innovations_algo_fast')
    arma_innovations_filter = getattr(
        _arma_innovations, prefix + 'arma_innovations_filter')

    # Run the innovations algorithm for ARMA coefficients
    arma_acovf = arima_process.arma_acovf(ar, ma,
                                          sigma2=sigma2, nobs=nobs) / sigma2
    acovf, acovf2 = arma_transformed_acovf_fast(ar, ma, arma_acovf)
    theta, v = arma_innovations_algo_fast(nobs, ar_params, ma_params,
                                          acovf, acovf2)
    v = np.array(v)
    if (np.any(v < 0) or
            not np.isfinite(theta).all() or
            not np.isfinite(v).all()):
        # This is defensive code that is hard to hit
        raise ValueError(NON_STATIONARY_ERROR)

    # Run the innovations filter across each series
    u = []
    for i in range(k_endog):
        u_i = np.array(arma_innovations_filter(endog[:, i], ar_params,
                                               ma_params, theta))
        u.append(u_i)
    u = np.vstack(u).T
    if normalize:
        u /= v[:, None]**0.5

    # Post-processing
    if squeezed:
        u = u.squeeze()

    return u, v



def arma_loglike(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute the log-likelihood of the given data assuming an ARMA process.
    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.
    Returns
    -------
    float
        The joint loglikelihood.
    """
    llf_obs = arma_loglikeobs(endog, ar_params=ar_params, ma_params=ma_params,
                              sigma2=sigma2, prefix=prefix)
    return np.sum(llf_obs)


def arma_loglikeobs(endog, ar_params=None, ma_params=None, sigma2=1,
                    prefix=None):
    """
    Compute the log-likelihood for each observation assuming an ARMA process.
    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.
    Returns
    -------
    ndarray
        Array of loglikelihood values for each observation.
    """
    endog = np.array(endog)
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]

    endog = np.ascontiguousarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = dtype(sigma2).item()

    func = getattr(_arma_innovations, prefix + 'arma_loglikeobs_fast')
    return func(endog, ar_params, ma_params, sigma2)



def arma_score(endog, ar_params=None, ma_params=None, sigma2=1,
               prefix=None):
    """
    Compute the score (gradient of the log-likelihood function).
    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.
    Returns
    ---------
    ndarray
        Score, evaluated at the given parameters.
    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglike(endog, params[:p], params[p:p + q], params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)



def arma_scoreobs(endog, ar_params=None, ma_params=None, sigma2=1,
                  prefix=None):
    """
    Compute the score (gradient) per observation.
    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.
    Returns
    ---------
    ndarray
        Score per observation, evaluated at the given parameters.
    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglikeobs(endog, params[:p], params[p:p + q],
                               params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)




def match(a, b):
     matching_values = set(a) & set(b)
     matching_indices = list() # or set() 
     for idx, value in enumerate(a):
         if value in matching_values:
             matching_indices.append(idx) # or add(idx) in case of a dict
     return matching_indices



def score_function_eval_hc(X_test, best_models, rho_vec, my_param_names, N_seg,N_mod):
    p_order = [best_models[i][0].order[0] for i in range(0,N_mod)]
    q_order = [best_models[i][0].order[2] for i in range(0,N_mod)]
    
    ar_part = [best_models[i][0].params()[0:p_order[i]] for i in range(0,N_mod)] 
    ma_part = [best_models[i][0].params()[(p_order[i]):(p_order[i]+q_order[i])] for i in range(0,N_mod)]
    sig_part = [best_models[i][0].params()[(p_order[i]+q_order[i]):] for i in range(0,N_mod)]
    
    FS_score = [[arma_scoreobs(X_test[i],ar_part[j], ma_part[j], sig_part[j] ) for i in range(0, N_seg)]for j in range(0,N_mod)]
    NEW_FS_score = [[FS_score[j][i][9:,]  for i in range(0, N_seg)] for j in range(0,N_mod)]

    FS_score_mean = [[np.mean(NEW_FS_score[j][i], axis=0) for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_sd = [[np.std(NEW_FS_score[j][i], axis=0) for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_cent  = [[(NEW_FS_score[j][i] - FS_score_mean[j][i])/FS_score_sd[j][i]  for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_cent2 =  [[np.sum(FS_score_cent[j][i], axis = 0)  for i in range(0, N_seg)] for j in range(0,N_mod)]
    
    FS_score_padded =  [[np.zeros(8) for i in range(0, N_seg)] for j in range(0,N_mod)]
    match_pos = [match(my_param_names, best_models[j][0].arima_res_.param_names)  for j in range(0,N_mod)]
    for j in range(0,N_mod):
     for i in range(0,N_seg):
        FS_score_padded[j][i][match_pos[j]] = FS_score_cent2[j][i] 
    
    FS_score_padded_w = [[ FS_score_padded[j][i] * rho_vec[:,1][j] for i in range(0, N_seg)] for j in range(0,N_mod)]
    return(FS_score_padded_w)




def score_function_eval_pd(X_test, best_models, rho_vec, my_param_names, N_seg,N_mod):
    intercept_part = [np.array([best_models[i][0].params()[0] ]) for i in range(0,N_mod)]    
    ar_part = [best_models[i][0].params()[1:4]  for i in range(0,N_mod)] 
    ma_part = [best_models[i][0].params()[4:7]  for i in range(0,N_mod)]
    sig_part = [best_models[i][0].params()[7]   for i in range(0,N_mod)]
    
    FS_score0 = [[arma_scoreobs(X_test[i],intercept_part[j], None, sig_part[j] ) for i in range(0, N_seg)]for j in range(0,N_mod)]
    FS_score = [[arma_scoreobs(X_test[i],ar_part[j], ma_part[j], sig_part[j] ) for i in range(0, N_seg)]for j in range(0,N_mod)]
    FS_score_fin = [[ np.c_[FS_score0[j][i][:,0], FS_score[j][i] ] for i in range(0, N_seg)]for j in range(0,N_mod)]
    NEW_FS_score = [[FS_score_fin[j][i][9:,]  for i in range(0, N_seg)] for j in range(0,N_mod)]
    
    FS_score_mean = [[np.mean(NEW_FS_score[j][i], axis=0) for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_sd = [[np.std(NEW_FS_score[j][i], axis=0) for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_cent  = [[(NEW_FS_score[j][i] - FS_score_mean[j][i])/FS_score_sd[j][i]  for i in range(0, N_seg)] for j in range(0,N_mod)]
    FS_score_cent2 =  [[np.sum(FS_score_cent[j][i], axis = 0)  for i in range(0, N_seg)] for j in range(0,N_mod)]

    
    FS_score_padded =  [[np.zeros(8) for i in range(0, N_seg)] for j in range(0,N_mod)]
    match_pos = [match(my_param_names, best_models[j][0].arima_res_.param_names)  for j in range(0,N_mod)]
    for j in range(0,N_mod):
     for i in range(0,N_seg):
        FS_score_padded[j][i][match_pos[j]] = FS_score_cent2[j][i] 
    
    FS_score_padded_w = [[ FS_score_padded[j][i] * rho_vec[j] for i in range(0, N_seg)] for j in range(0,N_mod)]
    return(FS_score_padded_w)





def calculate_K_GM_linear_no_centering_new(F_score_vec,N_seg):
    K = [ np.dot(np.vstack(F_score_vec[i]),np.vstack(F_score_vec[i]).T)  for i in range(0, N_seg)]
    K_id = [ ((np.trace(K[i])/8)*0.1 *np.eye(8)) + (0.9 * K[i] )  for i in range(0, N_seg)] 
    
    return(K_id)

def calculate_K_GM_linear_no_centering_noreg(F_score_vec,N_seg):
    K = [ np.dot(np.vstack(F_score_vec[i]),np.vstack(F_score_vec[i]).T)  for i in range(0, N_seg)]
    
    return(K)  




def GLRT_test(K0,K1,U0,U1):
    try:
        import numpy as np
        q1 = - np.dot(np.dot(U0.T , np.linalg.inv(K0) ), U0)
        q11 = q1 - ( np.linalg.slogdet(K0)[0] *  np.linalg.slogdet(K0)[1])
        q2 =   np.dot(np.dot(U1.T , np.linalg.inv(K1) ), U1) 
        q22 =  q2 + (np.linalg.slogdet(K1)[0] *  np.linalg.slogdet(K1)[1])
        glrt = q11 + q22
        return glrt         
    except np.linalg.LinAlgError as e:
     if 'Singular matrix' in str(e):
        return 'NAN'
    else:
        raise
        
        
        
        


#############
#  INDICES  #
#############

alfa_1 = 0.1
alfa_2 = 0.05
alfa_3 = 0.01

alfa = [alfa_1, alfa_2, alfa_3]

N_f  = 10

N_f_test = 89

ind = np.concatenate( (np.arange(0, 5000, 100), 5000), axis = None)


###############################
#      Set Directories        # 
###############################

mydir_read = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\New_Training_Testing_Sets\\'


###########################
# Loading the best models #
###########################


best_hc_models_f = np.load("C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M1\\Results_Fisher_Kernel\\Best_models\\best_hc_model_f.npy",
                           allow_pickle=True)

best_pd_models_f = np.load("C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M1\\Results_Fisher_Kernel\\Best_models\\best_pd_model_f.npy",
                           allow_pickle=True)

##########################################
#Loading the proportion of winning models#
##########################################


final_pd_order_rho_hc_f = np.load('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M1\\Results_Fisher_Kernel\\Best_models\\final_pd_order_rho_hc_f.npy',
                                  allow_pickle = True)
final_pd_order_rho_pd_f = np.load('C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M1\\Results_Fisher_Kernel\\Best_models\\final_pd_order_rho_pd_f.npy',
                                  allow_pickle = True)


#########################
#  LOAD TEST SEGMENTS   #
#########################

X_test_hc_f = np.load (mydir_read + 'X_test_hc_f.npy')

X_test_hc_f = np.load (mydir_read + 'X_test_pd_f_d1.npy')

X_test_hc_f_seg = [[ X_test_hc_f[j][ind[i]:ind[i+1]] for i in range(0, len(ind) -1) ] for j in range(0,N_f_test)]
X_test_hc_f_seg = [item for sublist in X_test_hc_f_seg for item in sublist]

N_f_test_batch = len(X_test_hc_f_seg)


############################
#centering the mini-batches#
############################

X_test_hc_f_seg_centr = [  ( X_test_hc_f_seg[i] - np.mean(X_test_hc_f_seg[i]) )/ np.std(X_test_hc_f_seg[i]) for i in range(0, N_f_test_batch)] 


##################################################################################
# Evaluate the model on the testing Segments & Evaluate the Fisher Score Vector  #
##################################################################################

N_mod = 10 # number of models
N_f_test_batch = 2000 #number of segs -->DELETE THIS LATER


my_param_names = ['intercept', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1', 'ma.L2', 'ma.L3', 'sigma2']


U_0_final = score_function_eval_hc(X_test_hc_f_seg_centr, best_hc_models_f,
                                   final_pd_order_rho_hc_f, my_param_names,
                                   N_f_test_batch, N_mod)

U_1_final = score_function_eval_pd(X_test_hc_f_seg_centr, best_pd_models_f,
                                   final_pd_order_rho_pd_f, my_param_names,
                                   N_f_test_batch, N_mod)


############################
# Aggregate them by model  #
############################

U_0_tilde = [sum(x) for x in zip(*U_0_final)] 
U_1_tilde = [sum(x) for x in zip(*U_1_final)] 

                   
##########################
# Compute Gram Matrices  #   we decide that I should not center but only regularise
##########################  


def calculate_K_GM_linear_no_centering_new(F_score_vec,N_seg):
    K = [ np.dot(np.vstack(F_score_vec[i]),np.vstack(F_score_vec[i]).T)  for i in range(0, N_seg)]
    K_id = [ ((np.trace(K[i])/8)*0.1 *np.eye(8)) + (0.9 * K[i] )  for i in range(0, N_seg)] 
    
    return(K_id)   
        
K_0_tilde_no_cent_noreg =  calculate_K_GM_linear_no_centering_noreg(U_0_tilde, N_f_test_batch)
K_1_tilde_no_cent_noreg =   calculate_K_GM_linear_no_centering_noreg(U_1_tilde, N_f_test_batch)
      
K0_new = calculate_K_GM_linear_no_centering_new(U_0_tilde, N_f_test_batch)
K1_new = calculate_K_GM_linear_no_centering_new(U_1_tilde, N_f_test_batch)  


###################################################
#  CHECK THE CONDITION NUMBER  & REGULARIZATIONs  #
###################################################


[1/(np.linalg.cond(K_0_tilde_no_cent_noreg[i], p = 2)) for i in range(0,100)]

[1/(np.linalg.cond(K_0_tilde_no_cent_noreg[i] + (np.eye(8) * (10**(-30))  )  , p = 2)) for i in range(0,100)]



#######################
# Compute GLRT Test   #
####################### 

GLRT_linear_4 = [GLRT_test(K0_new[i],K1_new[i],U_0_tilde[i], U_1_tilde[i]) for i in range(0,N_f_test_batch)]
GLRT_linear_4 


GLRT_linear_vec = np.vstack(GLRT_linear_4)
GLRT_linear_vec




