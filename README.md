# Pseudo_Pipeline

This repo contains all scripts necessary to generate mock pseudo-P(k) using a hacker version of [CAMB][1] for use as training or trial sets for a Gaussian process emulator, following the methodology of [Giblin et al. 2019][2]. The scripts required to do this are in PseudoEmulator and GPR_Emulator respectively, with the hacked CAMB scripts residing in CAMB-0.1.6.1.

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


### Editing the PseudoEmulator codes

In order to generate a trial set instead of a training set, the following variables should be changed in **LHC.py**, **Generate_Predictions.sh**, and **Plot_Predictions.py**:

 - **Output directory:** 'Training_Set' --> 'Trial_Set'
 - **Seed:** Seed=1 --> Seed=X (where X>1). This sets the random seed used by CAMB; use a different seed for training and trial sets. Typically I use Seed=1 for training sets and >=2 for trial sets.
 - **Data:** Data='Ultimate' --> Data='Ultimate12.5'. This variable defines the allowed range for cosmological params. If you add '12.5' to the variable name, it will prevent nodes from occupying the outer 12.5percent of the param space in any dimension (so all nodes fall in an inner subvolume). This is recommended for generating a trial set, because we're often only interested in how the emulation performs in the inner volume, not at the corners and edges where it is bound to fail. Note, 12.5 is arbitrary and any number less than 50 could be specified. Higher the number, the smaller the inner subvolume occupied by the trial nodes.
 - **Narrow_Range (LHC.py only):** 'Narrow_Range = False' --> 'Narrow_Range = True'. Closely connected to the **Data** variable defined above; you must change this in order for trial nodes to occupy an inner subvolume.
 - **Nodes:** 'Nodes=50' --> 'Nodes=X' (where X>50). This is arbitrary; typically I use 200 nodes for a trial set. One can also generate larger training sets imply by changing the Nodes variable and leaving all other variables mentioned here set to the training set values.


## PseudoOptimise

This directory contains two scripts for optimising the distribution of 13D input cosmologies. It uses the 50node Latin hypercube training set, generated inside in the **PseudoEmulator** directory following the instructions above as a starting point. It then adds nodes to this distribution according to two criteria:

 - **exploration:** where is the error from the emulator largest?
 - **exploitation:** a Gaussian shaped prior centred on the middle of the 13D param space, prioritising this volume.

After each node is added, the emulator is retrained. For more information, see [Rogers et al. 2018][3].

To run the optimisation, simply execute:

**python Pseudo_Nodes_Optimise.py**

The outputs are saved in, e.g.,: ../PseudoEmulator/Training_Set/Optimisation/Seed1Mx1.0_CPFinal_BFMixed-DataUltimate/nu0.19_stdevNone/

The functions & classes used in the optimisation can be found in optimisation.py. This is where one should turn if you want to modify the definition or exploitation, exploration, their sum (acquisition), or which optimiser routine is employed.

### Editing the PseudoOptimise codes

 - All of the variables defined in **PseudoEmulator** scripts mentioned above are repeated here. If you have changed any of those variables to, e.g., generate an initial training set with more nodes, you need to alter the variables in this script (Pseudo_Nodes_Optimise.py) as well.
 - The initial and final number of nodes used in the optimisation are defined by **Nodes_ini** and **Nodes_fin**. Note that **if you have just downloaded this repo, Nodes_fin is set to add just 2 nodes to the initial LHC** - this is a small number to test that the optimisation is working correctly for you. You should increase this; we have been working at adding 150 nodes to the initial 50, hence Nodes_fin=200.
 - **std_dev:** ~[0,1]. If finite, Gaussian random displacements are added to each optimised node, sampled from a normal distrn with this width. If None, no random displacement is added.
 - **nu:** This is a hyperparameter which weights the exploitation term; 0.19 is the default having been used in [Rogers et al. 2018][3]. A value of 1.0 means exploit & explor have equal weighting. 



[1]: https://camb.readthedocs.io/en/latest/
[2]: https://arxiv.org/abs/1906.02742
[3]: https://arxiv.org/abs/1812.04631