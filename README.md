# BSDMF
## Abstract

This paper focuses on fusing hyperspectral and multispectral images with an unknown arbitrary point spread function (PSF). Instead of obtaining the fused image based on the estimation of the PSF, a novel model is proposed without intervention of the PSF under Bayesian framework, in which the fused image is decomposed into double subspace-constrained matrix-factorization-based components and residuals. On the basis of the model, the fusion problem is cast as a minimum mean square error estimator of three factor matrices. Then, to approximate the posterior distribution of the unknowns efficiently, an estimation approach is developed based on variational Bayesian inference.  
Different from most previous works, the PSF is not required in the proposed model, and is not pre-assumed to be spatially-invariant. Hence, the proposed approach is not related to the estimation errors of the PSF, and has potential computational benefits when extended to spatially-variant imaging system.
Moreover, model parameters in our approach are less dependent on the input data sets and most of them can be learned automatically without manual intervention. Exhaustive experiments on three data sets verify that our approach shows excellent performance and more robustness to the noise with acceptable computational complexity, compared to other state-of-the-art methods.  

## Citation
@article{Lin2017Bayesian,
	title={Bayesian Hyperspectral and Multispectral Image Fusions via Double Matrix Factorization},  
	author={Lin, B. and Tao, X. and Xu, M. and Dong, L. and Lu, J.},  
	journal={IEEE Trans. Geosci. and Remote Sens.},  
	year={2017},   
	volume={55},   
	number={10},   
	pages={5666-5678},   
}
