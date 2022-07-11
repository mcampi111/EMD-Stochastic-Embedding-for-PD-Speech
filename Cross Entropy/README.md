# Optimal Time-Frequency EMD Partitions

In this folder, the class constructing an optimal partition of the time-frequency plane is provided. The RandomParition.py module can be imported and used to compute it. 
Instructions are given in the file as follows:

Class to estimate a partition of rectangle Pi = T x I for T = [t_0,t_N] and I = [omega_0,omega_M] into irregular
    grid of subrectangles given sample of points located in Pi.
    Estimation based on minmising Kullback–Leibler divergence between 
    two distributions: 
    -  true distribution defined by proportions of subrectangles to the whole area
    -  empirical sitribution defined by density of points in each of subrectangles
    Optimisation achieved by employing Cross Entropy Method (to cite) via 
    discret optimisation by multinomial distribution and continious optimisation
    via truncated normal distribution. Based on the paper titled
        "Stochastic Embedding of Empirical Mode Decomposition with Application in Parkinson's Disease Speech Diagnostics"
    
   The attributes are:
        points: 2D numpy array with first column points in T and second columns points in I
        M: int, number of partition to divide I
        D: int, number of partitions to divide T per each partition of I
        integrate_box_pdf: method to integrate pdf of points over subarea in Pi0 = [s,t]x[m,n]
                           for s < t and m< n, takes as arguments two list, [s,m] and [t,n]
        evaluate_pdf: method to evaluate pdf of a point in Pi, takes as argument [t,omega] in Pi