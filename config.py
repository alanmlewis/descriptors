#config.py
#lib
import os

#property, descriptor and model
property ='Boiling_point'
descriptor = 'Bit Morgan'
model= 'Random Forest Random Search'

#files to input
SMILES_file = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\compounds_Boiling point_InChI_SMILES.txt"
raw_properties_file =r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Just_Properties_with_ID\values_boiling_point"
#root_only = os.path.splitext(raw_properties_file)[0]## only want the root, changing the extentsion
#output_file = f'{root_only}_{property}.txt'
## need to configure: 
# property file 


## using YAML for config file as user friendly 
