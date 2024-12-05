# 29/08/2018, B. M. Giblin, PhD Student, Edinburgh
# Read in a Nodes files specified by Seed, CosmolPrior, BFs, Data, Node-Num, & Dimensions...
# ... and generate the CAMB predictions.
# SPEED = 1500 nodes (LCDM & NLCDM) PER HR

Seed=1
Mx="Optimal"
CPrior="Final"		# Range of cosmology nodes. MiraTitan is wider than Planck18
Basis="Mixed"		# Read these basis functions... 
Data="Final"		# ...and project them with these curves,... 
						# ...to get acceptable range of weights
Wiggles="True"

Nodes=16				
cosmol_dim=5
weight_dim=8
wiggle_dim=2
dimensions=$((cosmol_dim + weight_dim + wiggle_dim)) 

Redshift=(0.000)																	# USE THIS FOR A SINGLE REDSHIFT
#mapfile < /disk2/ps1/bengib/PseudoEmulator/Training_Set/Redshifts_0-2_41.dat		# USE THIS... 
#Redshift=${MAPFILE[*]}																# ...AND THIS FOR MULTIPLE REDSHIFTS

# !!! IMPORTANT LINE - WHERE THE OUTPUT IS GOING !!!
#DIR=/disk2/ps1/bengib/PseudoEmulator/Training_Set/TwoD_Grids/Dim0-2_Omm-H/CP${CPrior}_BF${Basis}-Data${Data}/Resolution50
#DIR=/disk2/ps1/bengib/PseudoEmulator/Training_Set/ 
DIR=/disk2/ps1/bengib/PseudoEmulator/Physical_Models/


CAMBDIR=/disk2/ps1/bengib/CAMB-0.1.6.1/
#pfile=$CAMBDIR/params.ini		#$CAMBDIR/params.ini # $CAMBDIR/GR_mnu_0p00_params.ini # I checked and these 2 param files give RESULTS CONSISTENT TO 0.6%
# Nodes file
#mapfile < $DIR/Nodes/Dim${dimensions}_Nodes_NoHeader.dat
#mapfile < $DIR/Nodes/Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_Nodes${Nodes}_Dim${dimensions}_NoHeader.dat 


# Make sure we ARE using physical parameterisation (we changed), and non-linear
sed -i 's/^use_physical .*$/use_physical = T/' $CAMBDIR/params.ini
sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $CAMBDIR/params.ini

start=$SECONDS
# Cycle through the Nodes
LOOP=$Nodes
for i in `seq 0 $LOOP`; do

	for z in ${Redshift[*]}; do

		echo "REDSHIFT $z ..."
		zlabel=`printf "%.3f" $z`
		OUTDIR=$DIR/Predictions/z$zlabel/
		if [ ! -d "$OUTDIR" ]; then
			mkdir $OUTDIR
		fi

		pfile=$CAMBDIR/params_Physical_BF${Basis}_ID${i}of${Nodes}.ini
		cp $CAMBDIR/params.ini $pfile

		if [ "$Wiggles" == "True" ]; then
			# Nodes File
			mapfile < $DIR/Nodes/z$zlabel/BF${Basis}_Nodes${Nodes}_Dim${dimensions}_w.dat
			var=$(echo "${MAPFILE[$i]}")	# Reads in a line
			# Outpute savename
			mapfile < $DIR/Nodes/Filenames_w.txt
			Filename=$(echo "${MAPFILE[$i]}")
		else
			# Nodes File
			mapfile < $DIR/Nodes/z$zlabel/BF${Basis}_Nodes${Nodes}_Dim${dimensions}_nw.dat
			var=$(echo "${MAPFILE[$i]}")	# Reads in a line
			# Outpute savename
			mapfile < $DIR/Nodes/Filenames_nw.txt
			Filename=$(echo "${MAPFILE[$i]}")
		fi


		var2=($var) 					# Turns line into an array with whitespace delimiter
		# Cosmological params	
		omm=$(echo "${var2[0]}")		
		omb=$(echo "${var2[1]}")
		h=$(echo "${var2[2]}")
		ns=$(echo "${var2[3]}")
		As=$(echo "${var2[4]}")

		omn=0.
		oml=$(expr 1.-$omm | bc)	# Lambda
		omc=$(expr $omm-$omb | bc)	# CDM
		# weight params
		w1=$(echo "${var2[5]}")
		w2=$(echo "${var2[6]}")
		w3=$(echo "${var2[7]}")	
		w4=$(echo "${var2[8]}")	
		w5=$(echo "${var2[9]}")
		w6=$(echo "${var2[10]}")
		w7=$(echo "${var2[11]}")
		w8=$(echo "${var2[12]}")

		echo "Generating CAMB prediction $i of $Nodes ..."
		echo "$omm $omb $h $ns $As"
	

		# Editing the params.ini 
		sed -i "s/^ombh2 .*$/ombh2 = $omb/" $pfile
		sed -i "s/^omch2 .*$/omch2 = $omc/" $pfile
		#sed -i "s/^omega_lambda .*$/omega_lambda = $oml/" $pfile	# Now determined by combo of other omegas
		sed -i "s/^omnuh2 .*$/omnuh2 = 0./" $pfile	# Adding BAO-wiggles manually with (a,b) params.
		sed -i "s/^hubble .*$/hubble = $h/" $pfile
		sed -i "s/^scalar_spectral_index(1) .*$/scalar_spectral_index(1) = $ns/" $pfile
		sed -i "s/^scalar_amp(1) .*$/scalar_amp(1) = $As/" $pfile


		Filename=$(echo "${MAPFILE[$i]}")	# Reads in a line
		sf=$DIR/Shapes/z$zlabel/${Filename}.dat
		sed -i "s#shapefile .*#shapefile = \'$sf\'#" $pfile


		sed -i "s/^transfer_redshift(1) .*$/transfer_redshift(1) = $zlabel/" $pfile				

		# First run CAMB with nlcdm shape NON-LINEAR
		cd $CAMBDIR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/NLPK_${Filename}_BF${Basis}.dat

		# Now with nlcdm shape LINEAR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 0/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/LPK_${Filename}_BF${Basis}.dat


		# Now run with vanilla LCDM ("V") LINEAR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = F/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/NLVK_${Filename}_BF${Basis}.dat
	done
	rm -f $pfile
done
cd $DIR

duration=$(( SECONDS - start ))
echo "It took $duration s to generate 2*$LOOP predictions"






