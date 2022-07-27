# System Model 2 (SM2): Gaussian Processses for IMFs.

This folder provides the code required to implement SM2 as described in the main body of the paper.

The Code folder is organized as follows:

1. **GLRT_Test_HC_PD_IMFs_male_parallel_Final.py** and **GLRT_Test_HC_PD_IMFs_female_parallel_Final.py**. This python file contains the code required for the GLRT testing procedure applied on SM2. There are multiple functions implemented which provide a package for this methodology, including the Fisher Kernel computation procedure required to obtain the Fisher score for the final Gram Matrix computed on the IMFs. See the paper for further details.
2. **S_M2_hc_female.py** and **S_M2_hc_male.py**. This python files provides the procedure for the fitting of the ARIMA models required for the derivation of the Fisher Kernel (healthy patients only - female and male) on the IMFs. For more details see the paper.
3. **S_M2_pd_female.py** and **S_M2_pd_male.py**. This python files provides the procedure for the fitting of the ARIMA models required for the derivation of the Fisher Kernel (sick patients only - female and male) on the IMFs. For more details see the paper.



