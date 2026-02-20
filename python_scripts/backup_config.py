#config.py
#libs
import os
from pathlib import Path

#property, descriptor and model

property = 'flashpoint'
##put _ between property words??
descriptor = ("Bit_Morgan",)
model = ("Random_Forest",)
optimization_algorithm = ("Grid_Search_optimizer")
##do you want graphs? 
compare_graphs = "yes"

## CAPITALISE EACH LETTER FOR START OF EACH WORD WHEN SELECTING MODEL

#files to input
root_dir = Path(__file__).resolve().parent.parent
#SMILES_file = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\compounds_Boiling point_InChI_SMILES_fixed.txt"
#SMILES_file = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\BP_InChI_SMILES_fixed.txt"
SMILES_file = root_dir / "datasets_and_SMILES"/ "flashpoint_compounds_InChI_SMILES.txt"
raw_properties_file =r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Just_Properties_with_ID\values_flashpoint"

#SMILES_file = root_dir / "datasets_and_SMILES"/ "BP_InChI_SMILES_fixed.txt"
#raw_properties_file = root_dir / "datasets_and_SMILES"/"values_boiling_point"
#SMILES_file = root_dir / "datasets_and_SMILES"/ "viscosity_compounds_InChI_SMILES.txt"
#raw_properties_file = root_dir / "datasets_and_SMILES"/"values_Viscosity_as_logVis"


## need to configure: 
# property file 


## using YAML for config file as user friendly?? 

