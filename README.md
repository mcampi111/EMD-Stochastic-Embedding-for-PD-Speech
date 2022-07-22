# Stochastic-Embedding-of-EMD-with-Application-in-Parkinson-s-Disease-Speech
This repository is linked to the methodology developed in the paper with title

**"Stochastic Embedding of Empirical Mode Decomposition with Application in Parkinson's Disease Speech Diagnostics"**. This paper is provided at this url https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3923615.

## **Abstract**

The time series method known as the Empirical Mode Decomposition (EMD) has become highly popular within different application areas due to its ability to adapt to non-stationary and non-linear time-frequency analysis settings more effectively than many other existing time series decomposition approaches. The EMD identifies a set of time domain basis functions, the Intrinsic Mode Functions (IMFs), which can also be transformed into a set of time varying frequency domain basis functions, the Instantaneous Frequencies (IFs). Like many time series decomposition methods, the EMD is considered an empirical path-wise decomposition approach applied to a time series realisation. However, to date, no statistical framework exists to encompass the study of this decomposition method from a stochastic representation or embedding perspective. This is essential to undertake statistical analysis tasks such as estimation and inference and to accurately incorporate statistical uncertainty quantification in out-of-sample predictions and forecasts (distinct from extrapolation). Hence, this work proposes a stochastic version of the EMD compatible with the path-wise features of the basis characteristics. The developed stochastic representations provide tractable statistical models, admitting flexible structures consistent with the characterising properties of the IMFs in their probabilistic representation when going from path-wise understanding to process embedding. In designing this novel stochastic embedding, we will focus on two examples of statistical EMD model constructions, which will be used to perform inference in detecting Parkinson's disease from speech signals. The proposed methods will be shown to outperform current state-of-the-art methods in Parkinson's disease based on speech analysis. This validates the exciting possibilities of developing a stochastic representation of the EMD decomposition methodology for time-frequency analysis.
  

## Contributions of the paper
The paper has multiple contributions, at both methodological and applied level:
1. A stochastic embedding model representation is developed for the Empirical Mode Decomposition (EMD) basis functions, known as the Intrinsic Mode Functions (IMFs), that is consistent with the characterising properties that the EMD requires for the IMFs. The focus of this stochastic representation will also be compatible with the setting in which the IMFs are characterised by statistical models comprised of B-spline and P-spline representations, as well as proposing flexible statistical models that readily lend themselves to estimation, inference and statistical forecasting methods for EMD decompositions.
2. To develop a family of statistical models for the proposed stochastic representation of EMD, a multi-kernel Gaussian Process framework is proposed. The particular features comprise a kernel construction suitable for modelling the non-stationary IMF basis GP spline representations. This uses a time series kernel representation based on a data-adaptive generative model embedding solution constructed via a Fisher Kernel. 
3. A second, localised stochastic solution, is also developed that defines an optimal set of band-limited basis functions stochastic model representations providing the following advantages: (1) one can focus on modelling specific bandwidths which might be significant for the application of interest; (2) one can formulate a set of basis functions whose marginal distributions are closer to a stationary distribution, compared to the original IMFs. Modelling the covariance function of such basis functions through a certain kernel is less challenging and will provide a more efficient solution for the MKL GP model representation.
4. The cross-entropy method is introduced in this context to find an optimal time-frequency partition which is fully data-adaptive. 
5. A novel solution to speech diagnostics for Parkinson's disease diagnostics and disease progression quantification is developed, which, when compared to state-of-the-art existing methods, is shown to be more sensitive and accurate for both the detection of early onset of Parkinson's disease as well as the quantification of disease progression. The solution is ultimately based on the stochastic EMD representations developed via the MKL GP model representations class.

## Motivations For Parkinson's Detection

## Organization of the Repository
The repository is organized in the following folders:

```diff
+ 1) Cross Entropy 
```
This folder is organized as follows:
1.  **figures**. This folder containes the figures produced in the notebook Example.ipynb and a gif showing the video of the optimization of the Cross-Entropy method.
2. **Example.ipynb**. This notebook provides a toy example showing how to use the package and the obtained results.
3.  **RandomPartition.py**. This is a python class which allows one to contruct an optimal partition of the time-frequency plane as presented in the main body of the paper. There are two options for the Importante Sampling distribution. One could pick a discrete distribution, i.e. a Multinomial, as provided in the paper. This is higly suggested as the one best performing in our experiments. Alternatively, one could also rely on a continuous distribution, i.e. a Truncated Normal. 
```diff
+ 2) Data 
```
This folder contains the data used in the experiments of the paper, the code used to extract the required features, i.e. IMFs and IFs, the notebooks used for the data preparation, including segmentation, pre-processing and cleaning. It is organized as follows:
1. **IMFs_IFs_extraction**. This folder  provides the R code required to extract the IMFs basis functions and the IFs, alias the Instantaneous Frequencies of the extracted IMFs (see the paper for more here).
2. **ReadText**. The speech signals used in the experiments are provided in this folder. Note that these are taken from https://zenodo.org/record/2867216#.Ytq343ZBzD4. These are speech recordings of healthy and sick individuals carried at King's College University by a mobile device. Two classes of sentences are provided: 1)the patients read a given text, and 2)the patients have a spontaneous dialogue with the interlocutor. In the paper, the first class only was considered for the analysis.
3. **Data_Anlysis.ipynb**. This notebook is to show an analisys of the provided data and the segments.
4. **Reshuffling_testing_sets.ipynb**. This notebook does the reshuffling since the training and testing sets have been made rotating
5. **Segmentation.ipynb**. This notebook carries the segmentation of the speech signals since the analysis is conducted over segments of the original speech signals.
6. **Training_Testing_segments.ipynb**. This notebook provides the preparation of training and testing sets.
7. **Training_Testing_segments_IMFs.ipynb**. This notebook  provides the preparation of training and testing sets for the IMFs extracted over the segments.
```diff
+ 3) System Model 1 (SM1)
```
This folder contains the code required to implement SM1 in the paper.  
Remark that for SM1 the Gaussian Process has to be calibrated on observations on the original signal interpolated through a cubic splie and denoted as $\tilde{s}(t)$ (for more details see the pdf of the paper), whose process is denoted as $\widetilde{S}(t)$. This represents the reference model to the EMD stochastic embeddings given in System Model 2 and System Model 3. Remark that under SM1, the GP model for  $S(t)$ is obtained via $$S(t) \stackrel{d}{=} \widetilde{S}(t) + \epsilon(t)$$ where, $\widetilde{S}(t)$ is treated as a GP $$\tilde{S}(t)  \sim \mathcal{GP} \left( \mu(t; \psi_{\widetilde{S}} ); k(t,t'; \theta_{\widetilde{S}} )\right)$$ with $\mu(t; \psi_{\widetilde{S}}  )$ and $k( t,t'; \theta_{\widetilde{S}}  )$ representing the mean and kernel functions respectively. The additive error $\epsilon(t)$ corresponds to a regression error based on using the spline representation $\widetilde{s}(t)$ for the representation and potentially calibration of the SM1. 
The folder is organized as follows:
1. **Notebooks**. This folder contains two notebooks: 
 - **Case_study_motivation_IMFs.ipynb**. This notebook aims to justify why a decomposition method dealing with non-stationarity and non-linearity of the data is highly required to analyse speech signals. The original speech signals' empirical covariance matrices are highly non-stationary; therefore, any standard method would not work in these settings. Moreover, the final goal is to detect fast changes signalling the presence of Parkinson's disease even at very early stages. Therefore, modelling the IMFs, i.e. the basis functions of the EMD, will be a lot more beneficial and provide more reliable and robust results to noise. Note that the second part of the notebook provides some early results conducted wiht fitting a Gaussian Process to the data using standard stationary kernel such as radial basis function (RBF), Square Exponential, etc. (see the following notebook for more information). The provided plots show results of the centered kernel target alignment (CKTA) (for more information about this measure see https://jmlr.org/papers/volume13/cortes12a/cortes12a.pdf) measuring the goodness of fit of these kernels with the data. The reader should not focus on this a lot, but the reasoning behind it is to show how the CKTA does not change across the x-axis providing different hyperparameters for the chosen kernel. Hence, this justifies the need for a more refined one, i.e. the Fisher kernel (see the paper for further details about this).
 - **Hyper_study.ipynb**. This notebook provides the kernel Gram Matrix of standard stationary kernels, which have been used in the second part of the above notebook. This is highly useful to understand how these standard structures will never be able to detect a highly complex structure as the one of the empirical covariances, particularly if fast changes must be detected to identify Parkinson's disease. 
3. **GLRT_Test_HC_PD_Final.py**. This python file contains the code required for the GLRT testing procedure applied on SM1. There are multiple functions implemented which provide a package for the testing procedure of this methodology, including the Fisher Kernel computation procedure required to obtain the Fisher score for the final Gram Matrix. We remark that the testing procedure is given as follows:
4. **S_M1_hc_final.py**.5
5. **S_M1_pd_final.py**.


```diff
+ 4) System Model 2 
```
This folder contains

```diff
+ 5) System Model 3 
```
This folder contains















