##importing all libraries
import os
import csv
import rdkit
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as AChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

import config
## print the configation just to check what im doing
print(config.property)
print(config.descriptor)
print(config.model)
#exit()

############################################
## convert prop files to csv function
def convert_to_csv(config, output_file = None):
	reading_file = pd.read_csv(config.raw_properties_file, sep=r"/s+")
	if output_file is None:
	    root_only = os.path.splitext(config.raw_properties_file)
        output_file = f'{root_only}_{property}.csv'
	reading_file.to_csv(output_file, index=False)
	print(f"Converted file to csv file called: {output_file}")
	config.raw_properties_file.close()
	return output_file
##issues: 
## raw prop file isn't defined for some reason

## extract properties from property files function
def extract_properties_from_csv(output_file):
	global property
	print(f"the property I'm extracting is {property}")
    ##exits function if you inputted the wrong property
    ask = input("is this correct?")
    #ask.evalaute()
    if ask == "no":
        exit()
    #if input("is this correct? please answer yes or no: ").lower() == "no":
      #exit()
	root_only_prop =os.path.splitext(output_file)[0]
	property_file = f"{root_only_prop}_{property}.csv"
    ##create file name for extracted properties
	final_property_file = os.path.join(os.path.dirname(output_file), output_file)
    ##take second column from file -- reader
    with open(output_file, "r", newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers= next(reader)
        properties = []
        for line_num, row in enumerate(reader, start=2):
            if len(row) < 2: ##check theres two columns 
                   raise ValueError(f"Row {line_num} only has 1 column: {row}")
            else: 
                   properties.append(row[1])
    ##save second column from old file to new file -- writer
    with open(final_property_file, 'w', newline='',encoding='utf-8') as outfile: 
        #like a pen needed to write
        writer= csv.writer(outfile)
        #writer.writerow([headers[1]])
        # commented out this -- i dont think its needed
        for value in properties: 
               writer.writerow([value])
               #this is me actually writing -- needed the pen first
        print(f"Extracted props saved to: {final_property_file}")
        return final_property_file
#issues: 
# need to define final_property_file, i think it'll be the just properties path
# indents with open need fixing -- idk why they arent working	

############################################
##change SMILES into a different descriptor (bit morgan fingerprints)
## DESCRIPTORS

def SMILES_to_BitMorgan(config):
    reading_txt_smiles = pd.read_fwf('compounds_Boiling point_InChI_SMILES.txt')
    reading_txt_smiles.to_csv('compounds_Boiling point_InChI_SMILES.csv')
    smiles_list = pd.read_csv('compounds_Boiling point_InChI_SMILES.csv')
    smiles_list = smiles_list[['SMILES']]

    # Fingerprint parameter

    fpgen = AChem.GetMorganGenerator(radius=3, fpSize=4096) # Larger fpSize means less chance of colliding bits

    # Generates the fingerprints and array

    mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list['SMILES']]	
    fpx = [fpgen.GetFingerprint(mols) for mols in mol]
    arr = np.array(fpx)

    # Prints fingerprints and the array

    for i, fingerprint in enumerate(fpx):
        print(f"Molecule {i+1} Fingerprint`: {fingerprint}")

    print(arr)
    print(arr.shape)

    # Exports the array to a csv

    df = pd.DataFrame(arr)
    bit_arrary_fingerprint = df.to_csv("bit_array_fingerprint.csv", header=False, index=False)
    return bit_array_fingerprint

# issues: 
#need to save the csv to somewhere, probably

#if config[descriptor] == 'Bit_Morgan': 
      ## run SMILES_Bit_Morgan function 

#############################################################
## MODELS
## train Ml model (here it's Random forest ) and test
def Random_forest_Random_search(config):
    pick_features = descriptor_file[config.descriptor]
    features = pd.read_csv(pick_features, header=None)
    dataset_RF = pd.read_csv(final_property_file)
    

    x = features
    y = dataset_RF

    model = RandomForestRegressor(random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=1))

    param_distributions = {
        'randomforestregressor__n_estimators': [200, 300, 400],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 3, 4, 5],
        'randomforestregressor__min_samples_leaf': [1, 2, 3, 4, 5]}

    random_search = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions = param_distributions,
        n_iter = 10,
        cv = 5)

    random_search.fit(x_train, y_train)
    y_predict = random_search.predict(x_test)

    print(f'Best score was using the following parameter set:')
    for key, val in random_search.best_params_.items():
        print(f'>> {key} = {val}')

    production_pipeline = random_search.best_estimator_
    production_pipeline.fit(x_train, y_train)

    y_best_predict = random_search.best_estimator_.predict(x_test)

    ##############################################################
    ## predict and output r2 and RMSE
    rmse = np.sqrt(root_mean_squared_error(y_test, y_best_predict))
    r2 = r2_score(y_test, y_best_predict)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    print(y_test[:20])
    print(y_best_predict[:20])



##pyhton dictionary 
descriptor_file = {
      "Bit Morgan": SMILES_to_BitMorgan
}

ML_models = {
      "Random Forest Random Search": Random_forest_Random_search
}
# running pre-processing functions 
pre_processing_function_1 = convert_to_csv()
pre_processing_function_2 = extract_properties_from_csv(pre_processing_function_1)
# exceution
##these become the new functions
descriptor_function = descriptor_file[config.descriptor]
model_function = ML_models[config.model]

##call the functions to execute
running_descriptor =descriptor_function()
running_model = model_function()