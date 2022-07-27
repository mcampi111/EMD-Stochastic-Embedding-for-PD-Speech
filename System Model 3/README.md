# System Model 3 (SM3): Gaussian Processses for BLIMFs.

This folder provides the code required to implement SM3 as described in the main body of the paper.

The Code folder is organized as follows:

1. **Notebook**. This folder contains: 
 - **CEM_real_data_example.ipynb**. This notebook contains an example of how the CEM has been used on the real speech dataset with all the steps starting from the IMFs, the extraction of the IFs and then the application of the CEM and the derivation of the BLIMFs.
 - **figures**. The figures produced in the above notebook are saved within this folder. Note that also the gifs showing the videos are provided.
2. **CE_parallal.py**. This python files run the CEM and computes the optimal partitions from the IFs.
3. **GLRT_Test_HC_PD_IMFs_Final.py**. This python file contains the code required for the GLRT testing procedure applied on SM3. There are multiple functions implemented which provide a package for this methodology, including the Fisher Kernel computation procedure required to obtain the Fisher score for the final Gram Matrix computed on the BLIMFs. See the paper for further details.
4. **IMF_BL_computation.py**. This python file extracts the BLIMFs from the IMFs, once the optimal partitions are computed trough the CEM.
5. **S_M3_final.py**. This python file provides the procedure for the fitting of the ARIMA models required for the derivation of the Fisher Kernel on the BLIMFs. For more details see the paper.
