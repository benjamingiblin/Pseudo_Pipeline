# 14/02/2019, B. M. Giblin, PhD Student, Edinburgh
# Scale the nodes for the physical models to match the training set

import numpy as np

# The following scaling assumes the Training set was produced with:
CPrior = "Final"			
Basis = "Mixed"				 
Data = "Ultimate"

cosmol_dim = 5
weight_dim = 8 		
wiggle_dim = 2
dimensions = cosmol_dim + weight_dim + wiggle_dim


# READ IN THE PHYSICAL NODES
z = "0.000"
Wiggles = "QuasiWiggles" # "w" or "nw" or "QuasiWiggles"
NumNodes = 16
Phys_Nodes = np.loadtxt('z%s/BF%s_Nodes%s_Dim%s_%s.dat'%(z,Basis,NumNodes,dimensions,Wiggles))

if CPrior == "Final":
	# Given to me by Matteo
	omega_m = [0.12, 0.155]							# !!! NOW little ommh^2 !!!
	omega_baryon = [0.0215, 0.0235]					# !!! NOW little ombh^2 !!!
	hubble = [60., 80.]
	n_s = [0.9, 1.05]
	A_s = np.exp(np.array([2.92, 3.16])) / 1e10			
Cosmol_Priors = np.vstack(( omega_m, omega_baryon, hubble, n_s, A_s ))		

if Data == "Ultimate":
	Weight_Priors = np.empty([weight_dim, 2])
	upp_bounds = [0., 1.5, 1.2, 0.75, 0.5, 0.25, 0.2, 0.1]
	low_bounds = [-20., -1.5, -1.2, -0.75, -0.5, -0.25, -0.2, -0.1]
	for i in range(weight_dim):
		Weight_Priors[i,:] = [ low_bounds[i], upp_bounds[i] ] 
	# Keep the priors as these if doing a LCDM run: we'll just manually set weight nodes to 0.

Wiggle_Priors = np.array([ [0.,1.], [-0.65, 0.1] ])
Priors = np.vstack(( Cosmol_Priors, Weight_Priors, Wiggle_Priors ))


Phys_Nodes_Scaled = np.zeros_like(Phys_Nodes)
for i in range(dimensions):
	Phys_Nodes_Scaled[:,i] = (Phys_Nodes[:,i] - Priors[i,0]) / abs(Priors[i,1] - Priors[i,0])
	if i >= cosmol_dim and i < cosmol_dim+weight_dim :
		Phys_Nodes_Scaled[:,i] = (Phys_Nodes[:,i] - Priors[cosmol_dim,0]) / abs(Priors[cosmol_dim,1] - Priors[cosmol_dim,0])
np.savetxt('z%s/BF%s_Nodes%s_Dim%s_%s_Scaled.dat'%(z,Basis,NumNodes,dimensions,Wiggles), Phys_Nodes_Scaled,fmt='%.7f')




