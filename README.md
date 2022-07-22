# Stochastic-Embedding-of-EMD-with-Application-in-Parkinson-s-Disease-Speech
This repository is linked to the methodology developed in the paper with title

**"Stochastic Embedding of Empirical Mode Decomposition with Application in Parkinson's Disease Speech Diagnostics"**. This paper is provided at this url https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3923615.

## **Abstract**

The time series method known as the Empirical Mode Decomposition (EMD) has become highly popular within different application areas due to its ability to adapt to non-stationary and non-linear time-frequency analysis settings more effectively than many other existing time series decomposition approaches. The EMD identifies a set of time domain basis functions, the Intrinsic Mode Functions (IMFs), which can also be transformed into a set of time varying frequency domain basis functions, the Instantaneous Frequencies (IFs). Like many time series decomposition methods, the EMD is considered an empirical path-wise decomposition approach applied to a time series realisation. However, to date, no statistical framework exists to encompass the study of this decomposition method from a stochastic representation or embedding perspective. This is essential to undertake statistical analysis tasks such as estimation and inference and to accurately incorporate statistical uncertainty quantification in out-of-sample predictions and forecasts (distinct from extrapolation). Hence, this work proposes a stochastic version of the EMD compatible with the path-wise features of the basis characteristics. The developed stochastic representations provide tractable statistical models, admitting flexible structures consistent with the characterising properties of the IMFs in their probabilistic representation when going from path-wise understanding to process embedding. In designing this novel stochastic embedding, we will focus on two examples of statistical EMD model constructions, which will be used to perform inference in detecting Parkinson's disease from speech signals. The proposed methods will be shown to outperform current state-of-the-art methods in Parkinson's disease based on speech analysis. This validates the exciting possibilities of developing a stochastic representation of the EMD decomposition methodology for time-frequency analysis.
  

## Contributions
The paper has multiple contributions, at both methodological and applied level:
1. A stochastic embedding model representation is developed for the Empirical Mode Decomposition (EMD) basis functions, known as the Intrinsic Mode Functions (IMFs), that is consistent with the characterising properties that the EMD requires for the IMFs. The focus of this stochastic representation will also be compatible with the setting in which the IMFs are characterised by statistical models comprised of B-spline and P-spline representations, as well as proposing flexible statistical models that readily lend themselves to estimation, inference and statistical forecasting methods for EMD decompositions.
2. To develop a family of statistical models for the proposed stochastic representation of EMD, a multi-kernel Gaussian Process framework is proposed. The particular features comprise a kernel construction suitable for modelling the non-stationary IMF basis GP spline representations. This uses a time series kernel representation based on a data-adaptive generative model embedding solution constructed via a Fisher Kernel. 
3. A second, localised stochastic solution, is also developed that defines an optimal set of band-limited basis functions stochastic model representations providing the following advantages: (1) one can focus on modelling specific bandwidths which might be significant for the application of interest; (2) one can formulate a set of basis functions whose marginal distributions are closer to a stationary distribution, compared to the original IMFs. Modelling the covariance function of such basis functions through a certain kernel is less challenging and will provide a more efficient solution for the MKL GP model representation.
4. The cross-entropy method is introduced in this context to find an optimal time-frequency partition which is fully data-adaptive. 
5. A novel solution to speech diagnostics for Parkinson's disease diagnostics and disease progression quantification is developed, which, when compared to state-of-the-art existing methods, is shown to be more sensitive and accurate for both the detection of early onset of Parkinson's disease as well as the quantification of disease progression. The solution is ultimately based on the stochastic EMD representations developed via the MKL GP model representations class.


## Organization of the Repository
The repository is organized in the following folders:
1. **Cross Entropy**. In this folder, the python class RandomPartition.py is provided. This class allows one to contruct an optimal partition of the time-frequency plane as presented in the main body of the paper. There are two options for the Importante Sampling distribution. One could pick a discrete distribution, i.e. a Multinomial, as provided in the paper. This is higly suggested as the one best performing in our experiments. Alternatively, one could also rely on a continuous distribution, i.e. a Truncated Normal. A notebook, i.e. Example.ipynb, with a toy example showing how to use the package and the obtained results is provided. A folder with the figures produced in the example and a gif showing a video of the optimization is also provided. 
2. **Data**. The speech signals used in the experiments are provided in this folder. Note that, these are taken from https://zenodo.org/record/2867216#.Ytq343ZBzD4. These are speech recordings of healthy and sick individuals carried at King's College University, by a mobile device. Two classes of sentence are provided: 1)the patients read a given text 2)the patients have a spontaneous dialogue with the interlocutor. In the paper, the first class only was considered for the analysis. These are provided in the folder named ReadText. The folder named IMFs_IFs_extraction instead provide both R and Python codes required to extract the IMFs basis functions and the IFs, alias the Instantaneous Frequencies of the extracted IMFs. 
3. **System Model 1**.
4. **System Model 2**.
5. **System Model 3**.













