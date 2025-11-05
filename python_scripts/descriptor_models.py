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

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


import config
## print the configation just to check what im doing
print(config.property)
print(config.descriptor)
print(config.model)
#exit()

############################################
## convert prop files to csv function
def convert_to_csv(config, output_file = None):
    reading_file = pd.read_csv(config.raw_properties_file, sep='\t', engine='python')
    if output_file is None:
        root_only = os.path.splitext(config.raw_properties_file)[0]
        output_file = f'{root_only}_{config.property}.csv'
    reading_file.to_csv(output_file, index=False)
    print(f"Converted file to csv file called: {output_file}")
    return output_file

## extract properties from property files function
def extract_properties_from_csv(config, output_file):
    print(f"the property I'm extracting is {config.property}")
    ##exits function if you inputted the wrong property
    ask = input("is this correct?")
    #ask.evalaute()
    if ask == "no":
        exit()
    #if input("is this correct? please answer yes or no: ").lower() == "no":
        #exit()
    root_only_prop =os.path.splitext(output_file)[0]
    property_file = f"{root_only_prop}_{config.property}.csv"
    ##create file name for extracted properties
    final_property_file = os.path.join(os.path.dirname(output_file), property_file)
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


############################################
##change SMILES into a different descriptor (bit morgan fingerprints)
## DESCRIPTORS

def SMILES_to_BitMorgan(config):
    # Read SMILES txt directly (one SMILES per line)
    smiles_list = pd.read_csv(config.SMILES_file, header=None, names=["SMILES"])
    smiles_list = smiles_list[['SMILES']]

    # Fingerprint parameter
    fpgen = AChem.GetMorganGenerator(radius=3, fpSize=4096)  # Larger fpSize = fewer collisions

    # Generate molecules
    mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list['SMILES']]

    # Generate fingerprints as bit vectors
    fpx = []
    for i, m in enumerate(mol, start=1):
        if m is None:
            print(f"Invalid SMILES at line {i}, writing NaN fingerprint")
            fpx.append(np.full(fpgen.GetFingerprint(Chem.MolFromSmiles("CC")).GetNumBits(), np.nan))
        else:
            bv = fpgen.GetFingerprint(m)
            arr_bits = np.zeros((bv.GetNumBits(),), dtype=int)
            for bit in bv.GetOnBits():
                arr_bits[bit] = 1
            fpx.append(arr_bits)

    arr = np.array(fpx)

    # Prints fingerprints and the array
    for i, fingerprint in enumerate(fpx):
        print(f"Molecule {i+1} Fingerprint: {fingerprint}")

    print(arr)
    print(arr.shape)

    # Exports the array to a csv
    df = pd.DataFrame(arr)
    bit_array_fingerprint = "bit_array_fingerprint.csv"
    df.to_csv(bit_array_fingerprint, header=False, index=False)

    print("SMILES to BitMorgan ran correctly")
    return bit_array_fingerprint
#############################################################
## spliting data

def split_test_train(config, final_property_file, running_descriptor): 
    print("Loading features from:", running_descriptor)
    features = pd.read_csv(running_descriptor, header=None)
    dataset_RF = pd.read_csv(final_property_file, header=None)
    

    x = features
    y = dataset_RF.values.ravel() ## flatten in 1D cause read_csv does a dataframe which is 2D 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
    print("It's been split!!")
    return x_test, x_train, y_test, y_train 
#############################################################
## MODELS
def Random_forest(config, final_property_file, running_descriptor, running_the_split):
    param_distributions = {
    'randomforestregressor__n_estimators': [500, 1000, 1500],
    'randomforestregressor__max_depth': [20, 40, 60, 80, None],
    'randomforestregressor__min_samples_split': [2, 5, 10],
    'randomforestregressor__min_samples_leaf': [1, 2, 3, 4],
    'randomforestregressor__max_features': ['sqrt', 'log2'] 
}
    
    model = RandomForestRegressor(random_state=1)
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=1))
    print("Random forest ran correctly!!")
    return pipeline, model, param_distributions

def SVR(config,final_property_file, running_desciptor, running_the_split): 
    print("Loading features from:", running_descriptor)

    # Define pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVR()
    )

    # Fit pipeline
    pipeline.fit(running_the_split.x_train, running_the_split.y_train.ravel())

    # Predict and evaluate --- not sure i actually need this bit now that i have optimization functions??
    y_predict_test = pipeline.predict(running_the_split.x_test)
    
    return pipeline, model

####################################################################################
## OPTIMIZATION ALGORITHMS 

def Random_Search(config, pipeline, model, param_distributions, x_train, x_test, y_train, y_test): 
    random_search = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions = param_distributions,
        n_iter = 50,
        cv = 5)

    random_search.fit(x_train, y_train)
    y_predict = random_search.predict(x_test)

    print(f'Best score was using the following parameter set:')
    for key, val in random_search.best_params_.items():
        print(f'>> {key} = {val}')

    #production_pipeline = random_search.best_estimator_
    #production_pipeline.fit(x_train, y_train)

    y_best_predict = random_search.best_estimator_.predict(x_test)

    ##############################################################
    ## predict and output r2 and RMSE
    rmse = np.sqrt(root_mean_squared_error(y_test, y_best_predict))
    r2 = r2_score(y_test, y_best_predict)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    print(y_test[:20])
    print(y_best_predict[:20])
    print("yay, we ran!!")
    return rmse, r2, y_best_predict

def Grid_Search(config, pipeline, model, param_distributions, x_train, x_test, y_train, y_test):
    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_distributions,
        n_jobs = -1,
        cv = 5)

    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    print(f'Best score was using the following parameter set:')
    for key, val in grid_search.best_params_.items():
         print(f'>> {key} = {val}')

    production_pipeline = grid_search.best_estimator_
    production_pipeline.fit(x_train, y_train)
    y_best_predict = production_pipeline.predict(x_test)

    rmse = np.sqrt(root_mean_squared_error(y_test, y_best_predict))
    r2 = r2_score(y_test, y_best_predict)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    print(y_test[:20])
    print(y_best_predict[:20])
    return r2, rmse, y_best_predict

def Successive_Halving(config, pipeline, model, x_train, x_test, y_train, y_test):
    param_grid = {
        'randomforestregressor__n_estimators': [200, 300, 400],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 3, 4, 5],
        'randomforestregressor__min_samples_leaf': [1, 2, 3, 4, 5]}

    halving_search = HalvingGridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        factor = 3,
        cv = 5)

    halving_search.fit(x_train, y_train)
    return_train_score = True
    y_predict = halving_search.predict(x_test)

    print(f'Best score was using the following parameter set:')
    for key, val in halving_search.best_params_.items():
        print(f'>> {key} = {val}')

    production_pipeline = halving_search.best_estimator_
    production_pipeline.fit(x_train, y_train)
    y_best_predict = production_pipeline.predict(x_test)

    rmse = np.sqrt(root_mean_squared_error(y_test, y_best_predict))
    r2 = r2_score(y_test, y_best_predict)

    print(f'Root Mean Squared Error: {rmse}')
    print(f'R2 Score: {r2}')

    print(y_test[:20])
    print(y_best_predict[:20])
    return r2, rmse, y_best_predict

##pyhton dictionary -- which functions correspond to what's in the config file
descriptor_file = {
      "Bit Morgan": SMILES_to_BitMorgan
}

ML_models = {
      "Random Forest": Random_forest,
      "SVR" : SVR

}

Optimization_algorithms = {
     "Random Search": Random_Search,
     "Grid Search": Grid_Search,
     "Successive Halving": Successive_Halving

}

## running pre-processing functions 
pre_processing_function_1 = convert_to_csv(config)
pre_processing_function_2 = extract_properties_from_csv(config, pre_processing_function_1)
# exceution
##these become the new functions
descriptor_function = descriptor_file[config.descriptor]
model_function = ML_models[config.model]
optimization_function = Optimization_algorithms[config.optimization_algorithm]


##call the functions to execute
running_descriptor =descriptor_function(config)
print("descriptor_function returned:", running_descriptor)

running_the_split = split_test_train(config, pre_processing_function_2, running_descriptor)

pipeline, model = model_function(config, pre_processing_function_2, running_descriptor, running_the_split)
r2, rmse, y_best_predict = optimization_function(config, pipeline, model)




# ## compare models
# def compare(rmse, r2): 
# ## make grpah 
# graph ==

# return graph

