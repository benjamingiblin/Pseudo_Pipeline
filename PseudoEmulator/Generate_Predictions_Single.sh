#!/bin/bash

# 29/08/2018, B. M. Giblin, PhD Student, Edinburgh
# Read in a Nodes files specified by Seed, CosmolPrior, BFs, Data, Node-Num, & Dimensions...
# ... and generate the CAMB predictions.
# SPEED = 1500 nodes (LCDM & NLCDM) PER HR

Redshift=(0.000) # 1.000)    # Use this for a few redshifts, or
# ...use this for a list of redshifts saved to file:
#mapfile < Training_Set/Redshifts_0-2_41.dat           

Nodes=1        # Number of nodes
cosmol_dim=5     # how many dimensions are LCDM 
weight_dim=8     # how many dim for the deviations from LCDM
wiggle_dim=0     # BAO wiggles (no longer used: keep as 0). 
dimensions=$((cosmol_dim + weight_dim + wiggle_dim)) 

# 27/05/2019: Want to make a training set optimised in 13D, NOT 15D.
# So introduce a new variable to facilitate this:
Nodes_Label=${Nodes} 

# !!! IMPORTANT LINE - WHERE THE OUTPUT IS GOING !!!
DIR=${PWD}/Training_Set
CAMBDIR=../CAMB-0.1.6.1
# hacked CAMB needs this library in path to work
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CAMBDIR}/lib

nodes_file=$1    # file containing the single proposal
mapfile < $nodes_file

OUTDIR="${nodes_file%/Nodes*}/"

# get the value of nu:
nu=${nodes_file#*nu}  # rm everything before 'nu'
nu=${nu%/Nodes*}           # rm everything after & incl. '/'

# Get the 'TAG' (Seed, Basis, Data, i.e. tagline this prediction corresponds to)
TAG=${nodes_file#*Optimisation/} # rm everything before (& incl) 'Optimisation/'
TAG=${TAG%/nu*}                  # rm everything after & incl '/nu'

# get node ID (e.g. 10of200):
whichnode=${nodes_file#*Node_Proposal_}
whichnode=${whichnode%.dat*}

# Set CAMB to use "physical parameterisation" (what form the LCDM params given are).
# and to predict the non-linear part of the power spectrum as best it can.
sed -i 's/^use_physical .*$/use_physical = T/' $CAMBDIR/params.ini
sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $CAMBDIR/params.ini

start=$SECONDS # take note of the time, so to time how long it takes.

# Cycle through the Nodes
LOOP=$((Nodes-1)) # Only need to loop through num of nodes -1 since we start at 0.
for i in `seq 0 $LOOP`; do
    
    # Load cosmol. params from file:
        pfile=$CAMBDIR/params_nu${nu}.dat
	
        # Make a copy of a blank input file to CAMB,
        # which you will then fill with the read in cosmol. params.
        cp $CAMBDIR/params.ini $pfile
	
        var=$(echo "${MAPFILE[$i]}")    # Reads in a line
        var2=($var)                                     # Turns line into an array with whitespace delimiter
        # Cosmological params   
        omm=$(echo "${var2[0]}")
        omb=$(echo "${var2[1]}")
        h=$(echo "${var2[2]}")
        ns=$(echo "${var2[3]}")
        As=$(echo "${var2[4]}")
	
        omn=0.
        oml=$(expr 1.-$omm | bc)        # Lambda
	omc=$(expr $omm-$omb | bc)      # CDM
	
        # weight params

        w1=$(echo "${var2[5]}")
        w2=$(echo "${var2[6]}")
        w3=$(echo "${var2[7]}") 
        w4=$(echo "${var2[8]}") 
        w5=$(echo "${var2[9]}")
        w6=$(echo "${var2[10]}")
        w7=$(echo "${var2[11]}")
        w8=$(echo "${var2[12]}")

        echo "Generating CAMB prediction $whichnode ..."
        echo "$omm $omb $h $ns $As"

        # Editing the CAMB input file: 
        sed -i "s/^ombh2 .*$/ombh2 = $omb/" $pfile
        sed -i "s/^omch2 .*$/omch2 = $omc/" $pfile
        #sed -i "s/^omega_lambda .*$/omega_lambda = $oml/" $pfile       # Now determined by combo of other omegas
        sed -i "s/^omnuh2 .*$/omnuh2 = 0./" $pfile      # Adding BAO-wiggles manually with (a,b) params.
        sed -i "s/^hubble .*$/hubble = $h/" $pfile
        sed -i "s/^scalar_spectral_index(1) .*$/scalar_spectral_index(1) = $ns/" $pfile
        sed -i "s/^scalar_amp(1) .*$/scalar_amp(1) = $As/" $pfile

        # Name of the output file for the matter pow. spec.:
        sf=$OUTDIR/Shapes/${TAG}_ID${whichnode}.dat 
        sed -i "s#shapefile .*#shapefile = \'$sf\'#" $pfile

        # Cycle through list of redshifts defined at the top:
	for z in ${Redshift[*]}; do

                echo "REDSHIFT $z ..."
                zlabel=`printf "%.3f" $z`
                OUTDIR_z=$OUTDIR/Predictions/z$zlabel/
                if [ ! -d "$OUTDIR_z" ]; then
                        mkdir $OUTDIR_z
                fi

                sed -i "s/^transfer_redshift(1) .*$/transfer_redshift(1) = $zlabel/" $pfile

                # First run CAMB to make pseudo-P(k) WITH NON-LINEAR
                cd $CAMBDIR
                sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $pfile
                sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
                ./camb $pfile
                mv $CAMBDIR/test_matterpower.dat $OUTDIR_z/NLPK_${TAG}_ID${whichnode}.dat

                # Now make it again WITHOUT NON-LINEAR correction on small scales.
                sed -i 's/^do_nonlinear .*$/do_nonlinear = 0/' $pfile
                sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
                ./camb $pfile
                mv $CAMBDIR/test_matterpower.dat $OUTDIR_z/LPK_${TAG}_ID${whichnode}.dat

                # Now run with vanilla LCDM ("V") LINEAR
                sed -i 's/^do_nonlinear .*$/do_nonlinear = 0/' $pfile
                sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = F/' $pfile
                ./camb $pfile
                mv $CAMBDIR/test_matterpower.dat $OUTDIR_z/LVK_${TAG}_ID${whichnode}.dat

                done
        rm -f $pfile
        done
cd $DIR

duration=$(( SECONDS - start ))
echo "It took $duration s to generate 2*$LOOP predictions"
