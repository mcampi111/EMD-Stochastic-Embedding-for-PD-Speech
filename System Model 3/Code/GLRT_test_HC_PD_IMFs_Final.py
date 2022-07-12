# -*- coding: utf-8 -*-
"""
@author: Marta
"""

#########################################
#   GLRT System Model 3  #    IMFs      #
#########################################
##########################
#     Female Patients    #
##########################

import glob
import os as os
import math
import numpy as np
#import multiprocess as mp   
import multiprocessing as mp # --> use this on the cluster
from pmdarima  import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics.pairwise import pairwise_kernels
import time
import sys  
import statsmodels.api as sm
from numpy.linalg import norm


from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from statsmodels.tsa.innovations import _arma_innovations



print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))


###############################
#      Set Directories        # 
###############################

#on Laptop
#mydir_read = 'C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\Data_extraction_&_preliminaries\\Results\\\IMF_BL_train_test\\Test_last_extracted\\'
#mydir_read1 = "C:\\Users\\Marta\\Desktop\\Parkinson_data\\Python_code\\System_M3\Results_Fisher_Kernel\\Best_models\\"


#On Cluster -- modify these
mydir_out = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Logs/Female_GLRT/SM3/'
mydir_read = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Data/IMFs_BL_last_extracted/'
mydir_read1 = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Data/'
mydir_res = '/lustre/home/ucabmc2/workspace/Python/EMD_GP_py/Results/Female_GLRT/SM3/'


###################################
#       Functions required        #   
###################################

NON_STATIONARY_ERROR = """\
The model's autoregressive parameters (ar_params) indicate that the process
 is non-stationary. The innovations algorithm cannot be used.
"""


def arma_innovations(endog, ar_params=None, ma_params=None, sigma2=1,
                     normalize=False, prefix=None, np = np, 
                     find_best_blas_type = find_best_blas_type, prefix_dtype_map = prefix_dtype_map,
                     _arma_innovations = _arma_innovations):
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
                    prefix=None, np = np, find_best_blas_type = find_best_blas_type,
                    prefix_dtype_map = prefix_dtype_map, _arma_innovations = _arma_innovations):
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
                  prefix=None, np = np, _get_epsilon = _get_epsilon, 
                  approx_fprime_cs = approx_fprime_cs, arma_loglikeobs = arma_loglikeobs):
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

    def func(params, np = np):
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



def score_function_eval_hc(X_test, best_models, rho_vec, my_param_names, N_seg,N_mod, arma_scoreobs = arma_scoreobs, 
                           match = match, np = np):
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



def score_function_eval_pd(X_test, best_models, rho_vec, my_param_names, N_seg,N_mod, arma_scoreobs = arma_scoreobs, 
                           match = match, np = np):
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
        



########################
#   Setting Indices    #
########################


status = [['hc']*19, ['pd']*12] 
gender = [['f']*18, 'm', ['f']*2, ['m']*10] 

status = [item for sublist in status for item in sublist]
gender = [item for sublist in gender for item in sublist]

n_patients = len(status)


patient_number = np.int32(sys.argv[3])

if gender[patient_number] == 'f':
    N_test = 89
    N_mod = 354
    ind_seg = np.concatenate( (np.arange(0, 4450, 1500), 4450), axis = None) #lenght 4 so 4-2 = 2
elif gender[patient_number] == 'm':
    N_test = 75
    N_mod = 298
    ind_seg = np.concatenate( (np.arange(0, 3750, 1350), 3750), axis = None) #lenght 4 so 4-2 = 2
    

ind = np.concatenate( (np.arange(0, 5000, 100), 5000), axis = None)

#ind_seg =  np.concatenate( (np.arange(0, 10, 5), 10), axis = None)
#ind1 = ind_seg[1]
#ind2 = ind_seg[2]
#ind_s = [ind1, ind2]
#ind_s


ind1 = np.int32(sys.argv[1])
ind2 = ind1 + 1

ind_s = [ind_seg[ind1], ind_seg[ind2]]


index_imf = np.int32(sys.argv[2])



##########################
#    Import Testing      #
##########################

IMFs_BL_files = os.listdir(mydir_read)

def order(elem):
    return elem.split("_")[3:6]



IMFs_BL_ordered = sorted(IMFs_BL_files, key=order) 

print(IMFs_BL_ordered)

IMF_BL_test = np.load( mydir_read + IMFs_BL_ordered[patient_number] , allow_pickle=True)

##############################################################
#       LOAD WEIGHTS RHO, BEST MODELS, TRAINING SEGMENTS     #
##############################################################

if gender[patient_number] == 'f':
    if index_imf == 0:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_f_BL_imf0.npy',
                                           allow_pickle= True)
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_f_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_f_imf0.npy',
                          allow_pickle=True)
        best_pd_model_imfs = np.load( mydir_read1 +'best_pd_model_BL_f_imf0.npy',
                          allow_pickle=True)
    elif index_imf == 1:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_f_BL_imf1.npy',
                                            allow_pickle= True)
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_f_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_f_imf1.npy',
                          allow_pickle=True)
        best_pd_model_imfs =  np.load( mydir_read1 +'best_pd_model_BL_f_imf1.npy',
                          allow_pickle=True)
    elif index_imf == 2:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_f_BL_imf2.npy',
                                            allow_pickle= True)
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_f_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_f_imf2.npy',
                          allow_pickle=True)
        best_pd_model_imfs = np.load( mydir_read1 +'best_pd_model_BL_f_imf2.npy',
                          allow_pickle=True)
    elif index_imf == 3:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_f_BL_imf3.npy',
                                            allow_pickle= True)  
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_f_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_f_imf3.npy',
                          allow_pickle=True)
        best_pd_model_imfs = np.load( mydir_read1 +'best_pd_model_BL_f_imf3.npy',
                          allow_pickle=True)
    elif index_imf == 4:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_f_BL_imf4.npy',
                                            allow_pickle= True)
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_f_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_f_imf4.npy',
                          allow_pickle=True)
        best_pd_model_imfs = np.load( mydir_read1 +'best_pd_model_BL_f_imf4.npy',
                          allow_pickle=True)

elif gender[patient_number] == 'm':
    if index_imf == 0:
        final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_m_BL_imf0.npy',
                                            allow_pickle= True)
        final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_m_BL_imfs.npy',
                                  allow_pickle = True)
        best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_m_imf0.npy',
                          allow_pickle=True)
        best_pd_model_imfs = np.load( mydir_read1 +'best_pd_model_BL_m_imf0.npy',
                          allow_pickle=True)
    elif index_imf == 1:
       final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_m_BL_imf1.npy',
                                            allow_pickle= True)
       final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_m_BL_imfs.npy',
                                  allow_pickle = True)
       best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_m_imf1.npy',
                          allow_pickle=True)
       best_pd_model_imfs =  np.load(mydir_read1 + 'best_pd_model_BL_m_imf1.npy',
                          allow_pickle=True)
    elif index_imf == 2:
       final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_m_BL_imf2.npy',
                                            allow_pickle= True)
       final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_m_BL_imfs.npy',
                                  allow_pickle = True)
       best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_m_imf2.npy',
                          allow_pickle=True)
       best_pd_model_imfs = np.load(mydir_read1 + 'best_pd_model_BL_m_imf2.npy',
                          allow_pickle=True)
    elif index_imf == 3:
       final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_m_BL_imf3.npy',
                                            allow_pickle= True)   
       final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_m_BL_imfs.npy',
                                  allow_pickle = True)
       best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_m_imf3.npy',
                          allow_pickle=True)
       best_pd_model_imfs = np.load(mydir_read1 + 'best_pd_model_BL_m_imf3.npy',
                          allow_pickle=True)
    elif index_imf == 4:
       final_pd_order_rho_IMF_hc = np.load(mydir_read1 + 'final_pd_order_rho_hc_m_BL_imf4.npy',
                                            allow_pickle= True)
       final_pd_order_rho_IMF_pd = np.load( mydir_read1 +'final_pd_order_rho_pd_m_BL_imfs.npy',
                                  allow_pickle = True)
       best_hc_model_imfs = np.load( mydir_read1 +'best_hc_model_BL_m_imf4.npy',
                          allow_pickle=True)
       best_pd_model_imfs = np.load(mydir_read1 + 'best_pd_model_BL_m_imf4.npy',
                          allow_pickle=True)
                      


IMF_test_seg = [[ IMF_BL_test[j][index_imf][ind[i]:ind[i+1]] for i in range(0, len(ind) -1) ] for j in range(0,N_test)] 
IMF_test_seg = [item for sublist in IMF_test_seg for item in sublist] 
IMF_test_seg = IMF_test_seg[ind_s[0] : ind_s[1]]

N_seg = len(IMF_test_seg)



############################
#centering the mini-batches#
############################

IMF_test_seg_centr = [   IMF_test_seg[i] - np.mean(IMF_test_seg[i])   for i in range(0, N_seg)] 


##################################################################################
# Evaluate the model on the testing Segments & Evaluate the Fisher Score Vector  #
##################################################################################

my_param_names = ['intercept', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1', 'ma.L2', 'ma.L3', 'sigma2']

# pool = mp.Pool(2)
# res_2 = pool.apply( score_function_eval_hc, args = (IMF_test_hc_f_seg_centr, best_hc_model_f_imfs, 
#                                                     final_pd_order_rho_IMF_hc_f, my_param_names,
#                                                     N_seg, N_mod, arma_scoreobs, match ) )



if __name__ == '__main__':    

     pool = mp.Pool(2)
     tic = time.perf_counter()
     U_0_final = pool.apply( score_function_eval_hc, args = (IMF_test_seg_centr, best_hc_model_imfs, final_pd_order_rho_IMF_hc,
                                                             my_param_names, N_seg, N_mod, arma_scoreobs, match) )
     U_1_final = pool.apply( score_function_eval_pd, args = (IMF_test_seg_centr, best_pd_model_imfs, final_pd_order_rho_IMF_pd,
                                                             my_param_names, N_seg, N_mod, arma_scoreobs, match) )
     toc = time.perf_counter()
     pool.close()    
     
print(f"Execute the computation of parallel fit in {toc - tic:0.4f} seconds")

############################
# Aggregate them by model  #
############################

U_0_tilde = [sum(x) for x in zip(*U_0_final)] 
U_1_tilde = [sum(x) for x in zip(*U_1_final)] 

##########################
# Compute Gram Matrices  #   
##########################  
        
K0_new = calculate_K_GM_linear_no_centering_new(U_0_tilde, N_seg)
K1_new = calculate_K_GM_linear_no_centering_new(U_1_tilde, N_seg)  


################
#  GLRT TEST   #
################

#all together -- one test per segment :
    
print("The evaluation of the GLRT has started")


GLRT_linear = [GLRT_test(K0_new[i],K1_new[i],U_0_tilde[i], U_1_tilde[i]) for i in range(0,N_seg)]
GLRT_linear
   
GLRT_linear_nosig = [GLRT_test(K0_new[i][0:7,0:7],K1_new[i][0:7,0:7],U_0_tilde[i][0:7], U_1_tilde[i][0:7]) for i in range(0,N_seg)]
GLRT_linear_nosig 
    

print("The evaluation of the GLRT has ended")


np.save(mydir_res +'GLRT_pat_' + str(patient_number) +  '_status_' + str(status[patient_number])  + '_' + str(gender[patient_number]) + "_imf_BL_" + str(index_imf) +  '_seg_'   + str(ind2) , GLRT_linear)
np.save(mydir_res +'GLRTnosig_pat_'  + str(patient_number) +  '_status_' + str(status[patient_number])  + '_' + str(gender[patient_number]) + "_imf_BL_" + str(index_imf) +   '_seg_'   + str(ind2)  , GLRT_linear_nosig)





