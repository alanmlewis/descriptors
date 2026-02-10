#!/bin/bash

models=("SVR" "Random_Forest" "XGBoost" "Lasso")
descriptors=("SMILES" "RDKit" "Bit Morgan")
optimization_algorithms=("Random_Search_optimizer" "Grid_Search_optimizer")

for model in "${models[@]}"; do
	for descriptor in "${descriptors[@]}"; do
		for optimization_algorithm in "${optimization_algorithms[@]}"; do
			sed -i "s/^[[:space:]]*model = .*/model = (\"$model\",)/" config.py
			
			sed -i "s/^[[:space:]]*descriptor = .*/descriptor = (\"$descriptor\",)/" config.py
	
	 		sed -i "s/^[[:space:]]*optimization_algorithm = .*/optimization_algorithm = (\"$optimization_algorithm\")/" config.py
			log_file="Visco_${optimization_algorithm}_${model}_${descriptor}_andshap.log"
			
			nohup python descriptor_models.py > "$log_file" 2>&1 &
			
			wait $!
			
			echo "completed all VISCO runs with model $model , $descriptor and $optimization_algorithm . Output saved to $log_file"
		
		done
	done
done
