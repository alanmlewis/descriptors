#!/bin/bash 
optimization_algorithms=("Random_Search_optimizer" "Grid_Search_optimizer")
descriptors=("SMILES" "RDKit" "Bit_Morgan")

for optimization_algorithm in "${optimization_algorithms[@]}"; do
	for descriptor in "${descriptors[@]}"; do
		sed -i "s/^[[:space:]]*descriptor = .*/descriptor = (\"$descriptor\",)/" backup_config.py

        	log_file="flash_${descriptor}_${optimization_algorithm}_compare_MODELS.log"

        	nohup python compare_models.py > "$log_file" 2>&1 &

        	wait $!

        	echo "completed all flash runs with models for $descriptor and ${optimization_algorithm} Output saved to $log_file"

	done
done
