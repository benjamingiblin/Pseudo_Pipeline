# Pseudo_Pipeline

This repo contains all scripts necessary to generate mock pseudo-P(k) using a hacker version of [CAMB][1] for use as training or test sets for a Gaussian process emulator, following the methodology of [Giblin et al. 2019][2]. The scripts required to do this are in PseudoEmulator and GPR_Emulator respectively, with the hacked CAMB scripts residing in CAMB-0.1.6.1.

Finally, the scripts in PseudoOptimise are used to optimise the distribution of input cosmological parameters (5 LCDM and 8 weights describing deviations from LCDM at scales of k>1 h/Mpc). The optimisation routine roughly follows the methodology of [Rogers et al. 2018][3].

## PseudoEmulator

cd into this directory before running any of the following scripts.

To generate a simple Latin hypercube (LHC) of input cosmological parameters:
**python LHC.py**
This makes an LHC of 50 nodes in 13 dimensions, with the dimensions being 5 LCDM parameters (Omega_mh^2, Omega_bh^2, h, n_s, A_s) followed by 8 weight parameters, defined via a principal component analysis, which cause deviations to the small scales whilst leaving the large-scales unchanged. The result is saved in the subdirectory, ./Training_Set/Nodes


Next you can generate the corresponding pseudo-P(k) predictions simple by running:
**./Generate_Predictions.sh**
This executes CAMB 3 times per node, in order to make the linear pseudo-P(k), the non-linear pseudo-P(k), and the linear LCDM P(k). The pseudo-P(k) are created by reading in a shape, saved in the ./Training_Set/Shapes, with the property that they are unity on large scales and deviate from unity on small (k>1 h/Mpc) scales, in a way which is described by the 8 weight params for this node. The shape is used to scale the initial P(k) in CAMB, creating a spectra which is the same/different from pure LCDM on large/small scales. The results are saved in the subdir, ./Training_Set/Predictions


You can then generate the log-boost-factors (log of the NL-pseudo-P(k) to Lin-pseudo-P(k) ratio) which are used to train the emulator. Do this by running:
**python Plot_Predictions.py**
This will make a plot of the linear and non-lin pseudo-P(k), plus the log boost factors, and save the boost factors in ./Training_Set/Predictions









[1]: https://camb.readthedocs.io/en/latest/
[2]: https://arxiv.org/abs/1906.02742
[3]: https://arxiv.org/abs/1812.04631