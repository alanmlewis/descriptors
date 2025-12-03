#config.py
#libs
import os
from pathlib import Path

#property, descriptor and model

property ='boiling_point'
descriptor = 'Bit Morgan'
model= 'Random Forest', 'CNN'
optimization_algorithm = "Random Search"

##do you want graphs? 
compare_graphs = "Yes".lower()

## CAPITALISE EACH LETTER FOR START OF EACH WORD WHEN SELECTING MODEL

#files to input
#SMILES_file = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\compounds_Boiling point_InChI_SMILES_fixed.txt"
#SMILES_file = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\BP_InChI_SMILES_fixed.txt"
#raw_properties_file =r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Just_Properties_with_ID\values_boiling_point"


root_dir = Path(__file__).resolve().parent.parent
SMILES_file = root_dir / "datasets_and_SMILES"/ "BP_InChI_SMILES_fixed.txt"
raw_properties_file = root_dir / "datasets_and_SMILES"/"values_boiling_point"


## need to configure: 
# property file 


## using YAML for config file as user friendly?? 

