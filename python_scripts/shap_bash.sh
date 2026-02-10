#!/bin/bash 

models=("SVR" "Random_Forest" "XGBoost" "Lasso")
descriptors=("SMILES" "RDKit" "Bit Morgan")
optimization_algorithms=("Random_Search_optimizer" "Grid_Search_optimizer")
properties=("boiling_point" "flashpoint" "Viscosity_as_logVis")

for property in "$properties[@}}"; do
	for model in "${models[@]}"; do
 	       for descriptor in "${descriptors[@]}"; do
		       sed -i "s/^[[:space:]]*property = .*/property = (\"$property\",)/" config.py
		       sed -i "s/^[[:space:]]*model = .*/model = (\"$model\",)/" config.py
	
        	       sed -i "s/^[[:space:]]*descriptor = .*/descriptor = (\"$descriptor\",)/" config.py

                       log_file="${property}_shap_${optimization_algorithm}_${model}_${descriptor}_V1.log"

                       nohup python shap_analysis.py > "$log_file" 2>&1 &

                       wait $!

                       echo "completed all $property runs with model $model , $descriptor and $optimization_algorithm . Output saved to $log_file"

                done
        done
done
