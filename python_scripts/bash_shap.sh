#!/bin/bash

descriptors=("SMILES" "RDKit" "Bit_Morgan")
models=("XGBoost" "SVR" "Lasso" "Random_Forest")
optimization_algorithms=("Grid_Search_optimizer" "Random_Search_optimizer")

for model in "${models[@]}"; do
	for descriptor in "${descriptors[@]}"; do
		for optimization_algorithm in "${optimization_algorithms[@]}"; do
			sed -i "s/^[[:space:]]*descriptor = .*/descriptor = (\"$descriptor\",)/" backup_config.py
			sed -i "s/^[[:space:]]*optmization_algorithm = .*/optimization_algorithm = (\"$optimization_algorithm\",)/" backup_config.py
			sed -i "s/^[[:space:]]*model = .*/model = (\"$model\",)/" backup_config.py
        		log_file="Visco_${descriptor}_${model}_${optimization_algorithm}_shap_pickle.log"

        		nohup python shap_analysis.py > "$log_file" 2>&1 &

        		wait $!

        		echo "completed all Visco runs with models for $descriptor , $model and $optimization_algorithm. Output saved to $log_file"
	
		done

	done
done
