#!/bin/bash 

descriptors=("SMILES" "RDKit" "Bit_Morgan")


for descriptor in "${descriptors[@]}"; do
	sed -i "s/^[[:space:]]*descriptor = .*/descriptor = (\"$descriptor\",)/" backup_config.py

        log_file="Visco_${descriptor}_compare_MODELS.log"

        nohup python compare_models.py > "$log_file" 2>&1 &

        wait $!

        echo "completed all Visco runs with models for $descriptor . Output saved to $log_file"
done
