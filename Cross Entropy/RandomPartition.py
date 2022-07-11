#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marta Campi
"""
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
sns.set_style("whitegrid")


class RandomPartition():
    '''Class to estimate a partition of rectangle Pi = T x I for T = [t_0,t_N] and I = [omega_0,omega_M] into irregular
    grid of subrectangles given sample of points located in Pi.
    Estimation based on minmising Kullback–Leibler divergence between 
    two distributions: 
    -  true distribution defined by proportions of subrectangles to the whole area
    -  empirical sitribution defined by density of points in each of subrectangles
    Optimisation achieved by employing Cross Entropy Method (to cite) via 
    discret optimisation by multinomial distribution and continious optimisation
    via truncated normal distribution. Based on:
        <to cite our paper>
    
    Attributes:
        points: 2D numpy array with first column points in T and second columns points in I
        M: int, number of partition to divide I
        D: int, number of partitions to divide T per each partition of I
        integrate_box_pdf: method to integrate pdf of points over subarea in Pi0 = [s,t]x[m,n]
                           for s < t and m< n, takes as arguments two list, [s,m] and [t,n]
        evaluate_pdf: method to evaluate pdf of a point in Pi, takes as argument [t,omega] in Pi
    '''



    def __init__(self,points,M,D,integrate_box_pdf,evaluate_pdf):
        
        self.M = M
        self.D = D
        self.points = points
        self.T = np.sort(np.unique(points[:,0]))
        self.omega_0 = min(points[:,1])
        self.omega_M = max(points[:,1])
        self.t_0 = min(points[:,0])
        self.t_N = max(points[:,0])
        self.delta_omega = self.omega_M - self.omega_0
        self.delta_t = self.t_N - self.t_0
        
        self._integrate_box_pdf = integrate_box_pdf
        self._evaluate_pdf = evaluate_pdf
        self._vec_truncate_sample = np.vectorize(self._truncate_sample)
    
       
    def initialize_partition(self,init_method = 'regular',seed = 1,figsize = (10,10),
                             plot_init = False,save_plot = False, plotpath = None,
                             plotname = 'init_partition.png'):
        
        if init_method == 'regular':
            self._initialize_regular()
        else:
            self.seed_init = seed
            self._initialize_random()            
        if plot_init:
            self.plot_partition(self.breaks_init,figsize,save_plot, plotpath,
                                plotname = 'partition_init.png')

    def _initialize_regular(self):     

        self.breaks_init = [np.linspace(start=self.omega_0, stop=self.omega_M, num=self.M+1)]
        self.breaks_init += [np.linspace(start=self.t_0, stop=self.t_N, num=self.D+1) for _ in range(self.M)]
        self.probs_init = [np.ones(self.M)/self.M]
        for m in range(self.M):
            self.probs_init.append(np.ones(self.D)/self.D)

        
    
    def _sample_proportions(self,size):
        
        proportions = self.random_state_init.dirichlet([3]*size, 1) 
        zero_proportions = np.where(proportions == 0)[0]
        for idx in zero_proportions:
            proportions[idx] = self.tol_init
        return np.insert(proportions/np.sum(proportions), 0, 0)
        
    def _initialize_random(self,tol_init = 1/10**3):        
        
        self.random_state_init = np.random.RandomState(self.seed_init)
        self.tol_init = tol_init         
        self.probs_init = self._sample_proportions(self.M)
        self.breaks_init = [self.omega_0 + self.delta_omega * np.cumsum(self.probs_init)]            
        for m in range(self.M): 
            self.probs_init.append(self._sample_proportions(self.D))
            self.breaks_init += [self.t_0 + self.delta_t * np.cumsum(self.probs_init[m+1])]    
    
    def _calculate_KL_kde(self,breaks):
        
        omegas = np.sort(breaks[0])
        KL = 0
        for m in range(self.M):
            s_m = np.sort(breaks[m+1])
            for d in range(self.D):
                size_Pi_m_d = (s_m[d+1] - s_m[d])* (omegas[m+1] - omegas[m])
                pdf_Pi_m_d =  self._integrate_box_pdf([s_m[d],omegas[m]],[s_m[d+1],omegas[m+1]])
                KL_m_d = size_Pi_m_d * np.log(size_Pi_m_d) - size_Pi_m_d * np.log(pdf_Pi_m_d)
                KL += KL_m_d
        KL /= self.size_Pi  
        KL += self.log_pdf_Pi - self.log_size_Pi  
        return KL
    
    def _convert_probs_to_breaks(self,probs):
        breaks = [self.omega_0 + np.insert(np.cumsum(probs[0]),0,0) * self.delta_omega]
        for m in range(self.M):
            breaks.append(self.t_0 + np.insert(np.cumsum(probs[m+1]),0,0) * self.delta_t)
        return breaks  
    
    def _calculate_KL_kde_multinomial(self,probs):
        breaks = self._convert_probs_to_breaks(probs)
        KL = 0
        for m in range(self.M):
            for d in range(self.D):
                size_Pi_m_d = (breaks[m+1][d+1] - breaks[m+1][d])* (breaks[0][m+1] - breaks[0][m])
                pdf_Pi_m_d = self._integrate_box_pdf([breaks[m+1][d],breaks[0][m]],[breaks[m+1][d+1],breaks[0][m+1]])
                KL_m_d = size_Pi_m_d * np.log(size_Pi_m_d) - size_Pi_m_d * np.log(pdf_Pi_m_d)
                KL += KL_m_d
        KL /= self.size_Pi  
        KL += self.log_pdf_Pi - self.log_size_Pi  
        return KL

    def _calculate_KL_kde_samples(self,samples):
        
        omegas = np.concatenate([[self.omega_0],np.sort(samples[0]),[self.omega_M]])
        KL = 0
        for m in range(self.M):
            s_m = np.concatenate([[self.t_0],np.sort(samples[m+1]),[self.t_N]])    
            for d in range(self.D):                        
                size_Pi_m_d = (s_m[d+1] - s_m[d]) * (omegas[m+1] - omegas[m])
                pdf_Pi_m_d = self._integrate_box_pdf([s_m[d],omegas[m]],[s_m[d+1],omegas[m+1]])
                KL_m_d = size_Pi_m_d * np.log(size_Pi_m_d) - size_Pi_m_d * np.log(pdf_Pi_m_d)
                KL += KL_m_d
        KL /= self.size_Pi  
        KL += self.log_pdf_Pi - self.log_size_Pi  
        return KL    
    
    def _truncate_sample(self, p, p_low, p_high, delta_p):
        return max(min(p,p_high - delta_p),p_low + delta_p)    
   
    def _generate_sample_multinom_batch(self,i):    
        sample_bins_omega = self.random_state.multinomial(self.N_grid_omega, self.probs_old[0])   
        zero_samples_omega = np.where(sample_bins_omega == 0)[0]
        for idx in zero_samples_omega:
            sample_bins_omega[idx] += self.tol_omega
        probs_new = [sample_bins_omega / np.sum(sample_bins_omega) ]      

        for m in range(self.M):
            sample_bins_t = self.random_state.multinomial(self.N_grid_t, self.probs_old[m+1])   
            zero_samples_t = np.where(sample_bins_t == 0)[0]
            for idx in zero_samples_t:
                sample_bins_t[idx] += self.tol_t
            probs_new.append( sample_bins_t /  np.sum(sample_bins_t) )   
        return probs_new
    
    def _generate_sample_multinom_parallel(self,i):
        np.random.seed(self.seed + i)  
        sample_bins_omega = np.random.multinomial(self.N_grid_omega, self.probs_old[0])   
        zero_samples_omega = np.where(sample_bins_omega == 0)[0]
        for idx in zero_samples_omega:
            sample_bins_omega[idx] += self.tol_omega
        probs_new = [sample_bins_omega / np.sum(sample_bins_omega) ]      

        for m in range(self.M):
            sample_bins_t = np.random.multinomial(self.N_grid_t, self.probs_old[m+1])   
            zero_samples_t = np.where(sample_bins_t == 0)[0]
            for idx in zero_samples_t:
                sample_bins_t[idx] += self.tol_t
            probs_new.append( sample_bins_t /  np.sum(sample_bins_t) )   
        return probs_new
    
    
    def _generate_sample_normal_parallel(self,i): 
        np.random.seed(self.seed + i)        
        omega_sim = np.random.multivariate_normal(self.breaks_old[0][1:self.M], self.cov_omegas)
        breaks_sim = [self._vec_truncate_sample(omega_sim,  self.omega_0,  self.omega_M , self.threshold_omega)]
        s_sim = [self._vec_truncate_sample(
            np.random.multivariate_normal(self.breaks_old[m+1][1:self.D],self.cov_t),
            self.t_0,  self.t_N ,self.threshold_t)  
                 for m in range(self.M)]
        return breaks_sim + s_sim
    
    def _generate_sample_normal_batch(self,i):      
        omega_sim = self.random_state.multivariate_normal(self.breaks_old[0][1:self.M], self.cov_omegas)
        breaks_sim = [self._vec_truncate_sample(omega_sim,  self.omega_0,  self.omega_M , self.threshold_omega)]
        s_sim = [self._vec_truncate_sample(
            self.random_state.multivariate_normal(self.breaks_old[m+1][1:self.D],self.cov_t),
            self.t_0,  self.t_N ,self.threshold_t)  
                 for m in range(self.M)]
        return breaks_sim + s_sim
    
    
    def estimate_partition_CEM_trun_normal(self,n_sim = 100, maxiter_CE = 100,tol_ce = 0.001, sigma2_omega = 0.1, sigma2_t = 0.1, 
                               a = 3,  beta = 0.6, rho = 0.2, seed = 1, seed_init = 1,init_method = None,bandwidth_kde = 0.1, 
                               parallel = False, n_workers = 4,sampler = 'batch', verbose = True,collect_it = True):
        self.seed = seed        
        if parallel:            
            pool = Pool(processes = n_workers)
            map0 = pool.map
            self._generate_sample_normal = self._generate_sample_normal_parallel 
        else:
            ### SIMULATING RANDOM NUMBERS IN PARALLEL IS TRICKY
            ### HERE'S ONE SOLUTION, WHEN SAMPLER IS NOT 'batch'
            ### SAMPLER 'batch' SHOULD BE USED ONLY IN NONPARALLEL MODE AS EACH WORKER THEN PRODUCES THE SAME SAMPLES
            ### POSSIBLE BETTER SOLUTION FOR PARALLEL MODE WITH SeedSequence FROM numpy
            ### https://numpy.org/doc/stable/reference/random/parallel.html
            if sampler == 'batch':
                # THIS SAMPLER MATCHES RESULTS CEM_normal_nonlinear_funs.ipy
                self.random_state = np.random.RandomState(seed)
                self._generate_sample_normal = self._generate_sample_normal_batch
            else:
                self._generate_sample_normal = self._generate_sample_normal_parallel                
            map0 = map        
    
        ########### PARAMETERS FOR TRUN. NORMAL SAMPLES        
        self.cov_omegas =  sigma2_omega * np.eye(self.M-1)
        self.threshold_omega = sigma2_omega * a
        self.cov_t = sigma2_t * np.eye(self.D-1)
        self.threshold_t = sigma2_t * a 

        ########### PARAMETERS FOR KL     
        self.size_Pi = self.delta_t * self.delta_omega  
        self.log_size_Pi = np.log(self.size_Pi)
        self.pdf_Pi = self._integrate_box_pdf([self.t_0,self.omega_0],[self.t_N,self.omega_M])
        self.log_pdf_Pi = np.log(self.pdf_Pi)

        ######### QUANTILE INDEX FOR CEM
        self.n_sim = n_sim
        range_n_sim = range(self.n_sim)
        rho_quantile_idx = int(rho * n_sim)

        ######### INIT OF PARAMS (BREAKS OF PARTITION) TO ESTIMATE
        init_time = datetime.now()
        print('Initializing partition: ',init_time.strftime("%H:%M:%S"))
        if not hasattr(self, 'breaks_init') or init_method is not None:
            if init_method is None:
                print('Default initialization: regular')
                init_method = 'regular'                
            self.initialize_partition(init_method = init_method,seed = seed_init)        

        self.i = 0 
        self.stop_cond = 0
        self.mu_old = self.breaks_init
        self.breaks_old = self.breaks_init
        self.KL_old = self._calculate_KL_kde(self.breaks_old) 
        if collect_it:
            self.breaks_list =[]
            self.KL_list= [] 

        print('CEM started: ',datetime.now().strftime("%H:%M:%S"))
        if verbose: 
            print('╔═══╦════════╦════════╦═══════╦═══════╦══════╦═════════╗')
            print('║ i ║ time   ║Gamma_k ║ minKL ║ maxKL ║ KL_i ║ rel_dif ║')

        while ((self.i < maxiter_CE) and (not self.stop_cond) ):
            current_time = datetime.now().strftime("%H:%M:%S")
            if collect_it:
                self.breaks_list.append(self.breaks_old)
                self.KL_list.append(self.KL_old)       
            
            samples = list(map0(self._generate_sample_normal, range_n_sim))
            KL_scores = list(map0(self._calculate_KL_kde_samples, samples))           
           
            scored_samples = list(zip(KL_scores, samples))
            scored_samples = sorted(scored_samples, key=lambda x: x[0])
            scored_samples_low = scored_samples[:(rho_quantile_idx+1)]        
            self.gamma_i = scored_samples_low[-1][0]

            self.mu_old = [sum([ll[1][0] for ll in scored_samples_low])/(rho_quantile_idx + 1)]
            omegas_update = np.concatenate([[self.omega_0],np.sort(self.mu_old[0]),[self.omega_M]])
            breaks_new = [(1-beta) * omegas_update  + beta * self.breaks_old[0]]

            for m in range(self.M):
                self.mu_old.append(sum([ll[1][m+1] for ll in scored_samples_low])/(rho_quantile_idx + 1))
                s_update = np.concatenate([[self.t_0],np.sort(self.mu_old[m+1]),[self.t_N]])
                breaks_new.append( (1-beta) * s_update  + beta * self.breaks_old[m+1])

            KL_new = self._calculate_KL_kde(breaks_new)
            self.KL_rel_diff = (self.KL_old - KL_new)/ self.KL_old
            
            if verbose:
                print('╠═══╬════════╬════════╬═══════╬═══════╬══════╬═════════╣')
                print('║{:3d}║{:s}║ {:3.3f}  ║ {:3.3f} ║ {:3.3f} ║ {:3.3f}║ {:3.3f} ║'.format(self.i,current_time,self.gamma_i, scored_samples[0][0], scored_samples[n_sim-1][0], self.KL_old, self.KL_rel_diff  * 100))
            
            if (self.KL_rel_diff > tol_ce) and (self.KL_rel_diff >= 0) :
                self.breaks_old = breaks_new
                self.KL_old = KL_new
                self.i += 1
            else:
                self.stop_cond = 1
                if self.KL_rel_diff < 0:  
                    print('Error! KL is increasing')
            self.seed += n_sim
        if parallel: 
            pool.close()
        end_time = datetime.now()
        print('CEM ended: ',end_time.strftime("%H:%M:%S"))
        print('Time: ',end_time - init_time)
        
    
    
    
    
    def estimate_partition_CEM_multinom(self,n_sim = 100, maxiter_CE = 100,tol_ce = 0.001,
                                        N_grid_omega = 100, N_grid_t = 100, tol_omega = 1, tol_t = 1,
                                        beta = 0.6, rho = 0.2, seed = 1, seed_init = 1,init_method = None,bandwidth_kde = 0.1, 
                               parallel = False, n_workers = 4,sampler = 'batch', verbose = True,collect_it = True):
        self.seed = seed        
        if parallel:            
            pool = Pool(processes = n_workers)
            map0 = pool.map
            self._generate_sample_multinom = self._generate_sample_multinom_parallel
        else:
            ### SIMULATING RANDOM NUMBERS IN PARALLEL IS TRICKY
            ### HERE'S ONE SOLUTION, WHEN SAMPLER IS NOT 'batch'
            ### SAMPLER 'batch' SHOULD BE USED ONLY IN NONPARALLEL MODE AS EACH WORKER PRODUCE THE SAME SAMPLES
            ### POSSIBLE BETTER SOLUTION FOR PARALLEL MODE WITH SeedSequence FROM numpy
            ### https://numpy.org/doc/stable/reference/random/parallel.html
            if sampler == 'batch':
                # THIS SAMPLER MATCHES RESULTS CEM_normal_nonlinear_funs.ipy
                self.random_state = np.random.RandomState(seed)
                self._generate_sample_multinom = self._generate_sample_multinom_batch
            else:
                self._generate_sample_multinom = self._generate_sample_multinom_parallel               
            map0 = map        
    
        ########### PARAMETERS FOR KL     
        self.size_Pi = self.delta_t * self.delta_omega  
        self.log_size_Pi = np.log(self.size_Pi)
        self.pdf_Pi = self._integrate_box_pdf([self.t_0,self.omega_0],[self.t_N,self.omega_M])
        self.log_pdf_Pi = np.log(self.pdf_Pi)

        ######### QUANTILE INDEX FOR CEM
        self.n_sim = n_sim
        range_n_sim = range(self.n_sim)
        rho_quantile_idx = int(rho * n_sim)

        ######### INIT OF PARAMS (BREAKS OF PARTITION) TO ESTIMATE
        init_time = datetime.now()
        print('Initializing partition: ',init_time.strftime("%H:%M:%S"))
        if not hasattr(self, 'breaks_init') or init_method is not None:
            if init_method is None:
                print('Default initialization: regular')
                init_method = 'regular'                
            self.initialize_partition(init_method = init_method,seed = seed_init)        

        self.i = 0 
        self.stop_cond = 0           
        
        
        self.N_grid_omega = N_grid_t
        self.N_grid_t = N_grid_t
        self.tol_omega = tol_omega
        self.tol_t = tol_t
        self.probs_old = self.probs_init
        self.breaks_old = self.breaks_init
        self.KL_old = self._calculate_KL_kde(self.breaks_old) 
        if collect_it:
            self.breaks_list =[]
            self.KL_list= [] 

        print('CEM started: ',datetime.now().strftime("%H:%M:%S"))
        if verbose: 
            print('╔═══╦════════╦════════╦═══════╦═══════╦══════╦═════════╗')
            print('║ i ║ time   ║Gamma_k ║ minKL ║ maxKL ║ KL_i ║ rel_dif ║')

        while ((self.i < maxiter_CE) and (not self.stop_cond) ):
            current_time = datetime.now().strftime("%H:%M:%S")
            if collect_it:
                self.breaks_list.append(self.breaks_old)
                self.KL_list.append(self.KL_old)       
            
            samples = list(map0(self._generate_sample_multinom, range_n_sim))
            KL_scores = list(map0(self._calculate_KL_kde_multinomial, samples))           
           
            scored_samples = list(zip(KL_scores, samples))
            scored_samples = sorted(scored_samples, key=lambda x: x[0])
            scored_samples_low = scored_samples[:(rho_quantile_idx+1)]        
            self.gamma_i = scored_samples_low[-1][0]

            probs_omega = sum([ll[1][0] for ll in scored_samples_low])/(rho_quantile_idx + 1)
            probs_new = [(1-beta) * probs_omega + beta * self.probs_old[0]]
            for m in range(self.M):
                probs_t = sum([ll[1][m+1] for ll in scored_samples_low])/(rho_quantile_idx + 1)
                probs_new.append( (1- beta) * probs_t + beta * self.probs_old[m+1])

            
            KL_new = self._calculate_KL_kde_multinomial(probs_new)
            self.KL_rel_diff = (self.KL_old - KL_new)/ self.KL_old
            
            if verbose:
                print('╠═══╬════════╬════════╬═══════╬═══════╬══════╬═════════╣')
                print('║{:3d}║{:s}║ {:3.3f}  ║ {:3.3f} ║ {:3.3f} ║ {:3.3f}║ {:3.3f} ║'.format(self.i,current_time,self.gamma_i, scored_samples[0][0], scored_samples[n_sim-1][0], self.KL_old, self.KL_rel_diff  * 100))
            
            if (self.KL_rel_diff > tol_ce) and (self.KL_rel_diff >= 0) :
                self.probs_old = probs_new
                self.breaks_old = self._convert_probs_to_breaks(self.probs_old)
                self.KL_old = KL_new
                self.i += 1
            else:
                self.stop_cond = 1
                if self.KL_rel_diff < 0:  
                    print('Error! KL is increasing')
            self.seed += n_sim
        if parallel: 
            pool.close()
        end_time = datetime.now()
        print('CEM ended: ',end_time.strftime("%H:%M:%S"))
        print('Time: ',end_time - init_time)
         
    def plot_KDE(self,figsize = (10,10),save_plot = False, plotpath = None,plotname = 'kde_points.png',n_mesh = 100j):
        
        tau_mesh,omega_mesh = np.mgrid[self.t_0:self.t_N:n_mesh,(self.omega_0):(self.omega_M):n_mesh]
        mesh_positions = np.vstack([tau_mesh.ravel(),omega_mesh.ravel()])
        Z =  np.reshape(self._evaluate_pdf(mesh_positions), omega_mesh.shape)
        
        fig = plt.figure(figsize=figsize)
        plt.plot(self.points[:,0],self.points[:,1],'b*', markersize=5)
        plt.imshow(np.rot90(Z), cmap= plt.cm.Reds, extent=[self.t_0,self.t_N,self.omega_0,self.omega_M])
        plt.ylim([self.omega_0-1, self.omega_M+1])
        plt.xlim([self.t_0, self.t_N])
        plt.xlabel('Time ' + r'$(t)$')
        plt.ylabel('Frequency ' + r'$(\omega)$')   
        if save_plot:
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            plt.savefig(plotpath + plotname)
        else:
            plt.show()    
    
    
    def plot_points(self,figsize = (10,3),save_plot = False, plotpath = None,plotname = 'points.png'):
        
        fig = plt.figure(figsize = figsize)
        plt.plot(self.points[:,0], self.points[:,1],'b*')
        plt.ylim([self.omega_0-1, self.omega_M+1])
        plt.xlim([self.t_0, self.t_N])
        plt.xlabel('Time ' + r'$(t)$')
        plt.ylabel('Frequency ' + r'$(\omega)$')   
        if save_plot:
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            plt.savefig(plotpath + plotname)
        else:
            plt.show()
    
    def plot_partition(self,breaks,figsize = (10,10),save_plot = False, plotpath = None,plotname = 'partition.png'):
        
        fig = plt.figure(figsize = figsize)
        plt.plot(self.points[:,0], self.points[:,1],'b*')
        plt.ylim([self.omega_0-1, self.omega_M+1])
        plt.xlim([self.t_0, self.t_N])
        plt.xlabel('Time ' + r'$(t)$')
        plt.ylabel('Frequency ' + r'$(\omega)$')        

        for m in range(self.M):
            plt.hlines(y=breaks[0][m],xmin = self.t_0, xmax = self.t_N)
            for d in range(self.D):
                plt.vlines(x=breaks[1+m][d],ymin = breaks[0][m], ymax = breaks[0][m+1])   
        plt.hlines(y=self.omega_M,xmin = self.t_0, xmax = self.t_N)    
        plt.vlines(x=self.t_N,ymin = self.omega_0, ymax = self.omega_M) 
        plt.vlines(x=self.t_0,ymin = self.omega_0, ymax = self.omega_M) 
        
        if save_plot:
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            plt.savefig(plotpath + plotname)
        else:
            plt.show  

    def animate_partition_fitting(self,figsize = (10,10),save_plot = False, plotpath = '',plotname = 'partition.gif'):
        
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
        ax.set_xlim(( self.t_0, self.t_N))
        ax.set_ylim((self.omega_0, self.omega_M)) 
        plt.close()
        def _animate_fun(i):
            omegas = self.breaks_list[i][0]
            s = self.breaks_list[i][1:]
            ax.clear()
            ax.plot(self.points[:,0], self.points[:,1],'b*')
            for m in range(self.M):
                ax.hlines(y=omegas[m],xmin = self.t_0, xmax = self.t_N)
                for d in range(self.D):
                        ax.vlines(x=s[m][d],ymin = omegas[m], ymax = omegas[m+1])   
            ax.hlines(y=self.omega_M,xmin = self.t_0, xmax = self.t_N)
            ax.vlines(x=self.t_N,ymin = self.omega_0, ymax = self.omega_M)
            ax.set_xlabel('Time ' + r'$(t)$')
            ax.set_ylabel('Frequency ' + r'$(\omega)$')
        anim = animation.FuncAnimation(fig, _animate_fun,frames=self.i,interval = 500)
        rc('animation', html='html5')
        
        if save_plot:
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            writergif = animation.PillowWriter(fps=1000)
            anim.save(plotpath + plotname, writer=writergif)
        else:
            return anim