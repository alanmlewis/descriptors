##importing all libraries
import os
import csv
import rdkit
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as AChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVR as SVRModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV as LassoCVmodel
from sklearn.linear_model import Lasso as LassoModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    #ask = input("is this correct?")
    #ask.evalaute()
    #if ask == "no":
    #    exit()
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

## DESCRIPTORS
def RDkit_Descriptors(config): 
    smiles_list = pd.read_csv(config.SMILES_file, header=None, names=["SMILES"])
    smiles_list = smiles_list[['SMILES']]

    mols = []
    for smiles in smiles_list['SMILES']: 
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: 
            mols.append(mol)
        else: 
            print(f"Invalid smiles skipped:{smiles}")

    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    calc = MolecularDescriptorCalculator(descriptor_names)
    
    ##Calculate descriptors
    descriptor_values = []
    for mol in mols:
        try:
            descriptor_values.append(calc.CalcDescriptors(mol))
        except:
            ##if molecule fails descriptor calculation
            descriptor_values.append([None]*len(descriptor_names))
            print(f"theres been an error with:{mol}")
    arr = np.array(descriptor_values)

    # #export to csv
    df = pd.DataFrame(arr, columns=descriptor_names)
    descriptor_csv = "rdkit_descriptors.csv"
    df.to_csv(descriptor_csv, header=False, index=False)

    print(arr.shape)
    print("RDKit descriptors ran correctly")
    return descriptor_csv

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

def SMILES_to_Matrices(config): 
    df = pd.read_csv(config.SMILES_file, header=None, names=['SMILES'])

## Convert column to a list of strings and remove anything unwanted
    smiles_list = df['SMILES'].dropna().astype(str).str.strip().str.replace(r'\s+', '', regex=True).tolist()

## set vocab up and enumerate (i.e. give each character a numerical value corresponding to position in list)
    vocab = ('#', '%', "(", ")",'@','+','-','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', "[", "]", 'c', 'i', 'l', 'n',"o", 'r', 's','/', "\\",".")
    char_to_ind = {char: ind for ind, char in enumerate(vocab)}
## define the matrix as N x L where N is charcters in vocab and L is the length of smiles string
    vocab_size=len(vocab)

    max_smiles_length= max(len(smiles)for smiles in smiles_list) 
## flattened matrices
    vectors = [] 
    for smiles in smiles_list:
            smiles_length= len(smiles)
            one_hot_matrix = np.zeros((smiles_length,vocab_size), dtype=int) ## make a massive matrix of zeros
            for i, char in enumerate(smiles):
                if char in char_to_ind:
            ## using char_to_ind as a dictionary
                    one_hot_matrix[i, char_to_ind[char]]= 1
                else: 
                ## for every value in smiles, if it is also in vocab change the 0 to a 1. If the value in smiles 
    ## is non existent in vocab, throw an error and say what the character was. 
                    raise ValueError("the character",char,"does not exist in the vocabulary")
        ## make padded matrices for consistency
            padded_matrix= np.pad(one_hot_matrix,((0,max_smiles_length - smiles_length),(0, 0)), mode='constant',constant_values=0)
        ## flatten matrix into vector for algorithm to read it
            flattened_matrix= padded_matrix.flatten()
            #flattened_string = ''.join(map(str, flattened_matrix))
            #smile_string_with_label = flattened_string + ' '
            #vectors.append(smile_string_with_label)
            vectors.append(flattened_matrix)
    temp_df = pd.DataFrame(vectors) 
    SMILES_matrices_csv = "SMILES_matrices.csv"
    temp_df.to_csv(SMILES_matrices_csv, header=False, index=False)
    print(f"flattened one hot matrices all saved without SMILES lables and seperators added.")
    return SMILES_matrices_csv
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
def Random_forest(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train):
    param_distributions = {
    'randomforestregressor__n_estimators': [500, 1000, 1500],
    'randomforestregressor__max_depth': [20, 40, 60, 80, None],
    'randomforestregressor__min_samples_split': [2, 5, 10],
    'randomforestregressor__min_samples_leaf': [1, 2, 3, 4],
    'randomforestregressor__max_features': ['sqrt', 'log2'] 
}
    
    #model = RandomForestRegressor(random_state=1)
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=1))
    print("Random forest ran correctly!!")
    return pipeline, param_distributions

def SVR(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train): 
    print("Loading features from:", running_descriptor)

    # Define pipeline
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        SVRModel()
    )

    # Fit pipeline
    #pipeline.fit(running_the_split.x_train, running_the_split.y_train.ravel())
    # Predict and evaluate --- not sure i actually need this bit now that i have optimization functions??
    #y_predict_test = pipeline.predict(running_the_split.x_test)
    
    param_distributions = {
                    "svr__C": np.logspace(-2, 3, 20),
                    "svr__gamma": ["scale", "auto"] + list(np.logspace(-4, 1, 10)),
                    "svr__epsilon": np.logspace(-3, 0, 10),
                    "svr__kernel": ["rbf"]
                    }
    return pipeline, param_distributions

def XGBoost(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train):
    ## Initialize XGBoost Regressor
    print("loading feaure from:", running_descriptor)
    pipeline = XGBRegressor()
    param_distributions = { "n_estimators": [100, 200, 400],        ## number of trees
        "learning_rate": [0.01, 0.05],      ## shrinkage step size
        "max_depth": [2, 3, 4],             ##  tree depth
        "subsample": [0.6, 0.7],           ## fraction of data to sample per tree
        "colsample_bytree": [0.6, 0.7],## fraction of features to sample per tree
        "min_child_weight": [1, 5, 10], ## restricts how deep a tree grows
        "reg_lambda": [1, 5, 10], ##penalizes sum of squared leaf weights
        "reg_alpha": [0, 0.5, 1], ## penalty, simplfies model 
        "gamma": [0, 0.1, 0.5],## minimum loss reduction required to make a split on leaf node
        "random_state": [42],
        "n_jobs": [8],  ## number of CPU cores im using
        }
    return pipeline, param_distributions

def LassoCV(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train):
    print("loading features from:", running_descriptor)
    pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LassoCVmodel(max_iter=2000) ## may want to up this to 10000 for morgan
            )
    param_distributions = {"alphas":[0.001, 0.01, 0.1, 1, 10],
            "random_state": [42],
            "n_jobs": [5]
        }
    return pipeline, param_distributions

def Lasso(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train):
    print("loading feature from:", running_descriptor)
    pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SelectKBest(f_regression),
            LassoModel(max_iter=2000)
                        )
    K_values = [2, 4, 6, 8, 10]
    param_distributions = {
            "lasso__alpha": np.logspace(-1, 2, 50),
            "selectkbest__k": K_values
            }
    return pipeline, param_distributions

##park this for now
def CNN(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train): 
    ##importing stuff
    import torch 
    import torch.nn as nn 
    import torch.nn.functional as F 
    from skorch import NeuralNetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    ## processing into 3D for CNN: convert to NumPy float32
#    x_train_np = x_train.values.astype(np.float32)
#   x_test_np  = x_test.values.astype(np.float32)

    ##Add channel dimension
#   x_train_tensor = torch.from_numpy(x_train_np[:, None, :]).float()
#   x_test_tensor  = torch.from_numpy(x_test_np[:, None, :]).float()

    ##y as float32 tensors
#   y_train_tensor = torch.from_numpy(y_train.astype(np.float32))[:, None].float()
#    y_test_tensor  = torch.from_numpy(y_test.astype(np.float32))[:, None].float()

    input_dim = x_train.shape[1] ##number of columns in the dataframe
    
    class CNNregression(nn.Module): ## parent class
        def __init__(self, input_dim): 
            super(CNNregression, self).__init__() ## telling it it's a model
            self.conv1  = nn.Conv1d(1, 32, kernel_size =8) ## layer 1 
            self.conv2 = nn.Conv1d(32, 64, kernel_size =8) ## layer 2

            conv_out_dim = input_dim - 7 
            conv_out_dim = conv_out_dim - 7 ## computes length of sequence after each layer

            self.fc1 = nn.Linear(64 * conv_out_dim, 128 ) ## fully connected dense layer
            self.fc2 = nn.Linear(128, 1) ## output predicts 1 value 
            
        def forward(self, x): 

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = NeuralNetRegressor(
        CNNregression,
        module__input_dim=input_dim,
        max_epochs=20,
        lr=0.001,
        optimizer=torch.optim.Adam,
        batch_size=32,
        iterator_train__shuffle=True,
    )

    ##create a pipeline (scaler to neural net)
    pipeline = make_pipeline(
        StandardScaler(),
        net
    )

    param_distributions = {
        'neuralnetregressor__lr': [0.0001, 0.001, 0.01],
        'neuralnetregressor__max_epochs': [20, 50, 100],
        'neuralnetregressor__batch_size': [16, 32, 64],
        'neuralnetregressor__optimizer': [
            torch.optim.Adam,
            torch.optim.RMSprop
        ]
    }

    return pipeline, param_distributions 
        ######## THIS MODEL IS NOT FINISHED 



###########################################################################

# OPTIMIZATION ALGORITHMS

def Random_Search(config, pipeline, param_distributions, x_test, x_train, y_test, y_train): 
    random_search = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions = param_distributions,
        n_iter = 50,
        cv = 5,
        n_jobs = 4)

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
    rmse = root_mean_squared_error(y_test, y_best_predict)
    r2 = r2_score(y_test, y_best_predict)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    print(y_test[:20])
    print(y_best_predict[:20])
    print("yay, we ran!!")
    optimizer_object = random_search
    return  r2, rmse, y_best_predict, optimizer_object

def Grid_Search(config, pipeline, param_distributions, x_test, x_train, y_test, y_train):
    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_distributions,
        n_jobs = 10,
        cv = 5)

    grid_search.fit(x_train, y_train)
    y_predict = grid_search.predict(x_test)

    print(f'Best score was using the following parameter set:')
    for key, val in grid_search.best_params_.items():
         print(f'>> {key} = {val}')

    production_pipeline = grid_search.best_estimator_
    production_pipeline.fit(x_train, y_train)
    y_best_predict = production_pipeline.predict(x_test)

    rmse = root_mean_squared_error(y_test, y_best_predict)
    r2 = r2_score(y_test, y_best_predict)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')
    print("Train R2:", r2_score(y_train, production_pipeline.predict(x_train)))
    print("best alpha:", grid_search.best_params_)
    print(y_test[:20])
    print(y_best_predict[:20])
    optimizer_object = grid_search
    return r2, rmse, y_best_predict, optimizer_object


## ignore this thing
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
      "Bit Morgan": SMILES_to_BitMorgan, 
      "RDKit": RDkit_Descriptors,
      "SMILES": SMILES_to_Matrices
}

ML_models = {
      "Random_Forest": Random_forest,
      "SVR" : SVR,
      "XGBoost": XGBoost,
      "LassoCV": LassoCV,
      "CNN": CNN,
      "Lasso": Lasso
}

Optimization_algorithms = {
     "Random_Search_optimizer": Random_Search,
     "Grid_Search_optimizer": Grid_Search,
     "Successive_Halving_optimizer": Successive_Halving

}
## running pre-processing functions 
pre_processing_function_1 = convert_to_csv(config)
pre_processing_function_2 = extract_properties_from_csv(config, pre_processing_function_1)

for desc in config.descriptor:
    descriptor_function = descriptor_file[desc]

##call the functions to execute -- this system will work for multiple pairs of opts and models 
    running_descriptor =descriptor_function(config)
    print("descriptor_function returned:", running_descriptor)
    results= []

    for model_name in config.model:
        print(f"Current model running: {model_name}")
        model_function = ML_models[model_name]
        x_test, x_train, y_test, y_train = split_test_train(config, pre_processing_function_2, running_descriptor)

        #if model_name == "CNN":
         ## processing into 3D for CNN: convert to NumPy float32
            #x_train_np = x_train.values.astype(np.float32)
            #x_test_np  = x_test.values.astype(np.float32)

    ##Add channel dimension
            #x_train_tensor = torch.from_numpy(x_train_np[:, None, :]).float()
            #x_test_tensor  = torch.from_numpy(x_test_np[:, None, :]).float()

    ##y as float32 tensors
            #y_train_tensor = torch.from_numpy(y_train.astype(np.float32))[:, None].float()
            #y_test_tensor  = torch.from_numpy(y_test.astype(np.float32))[:, None].float()
            #x_train = x_train_tensor
            #x_test = x_test_tensor
            #y_train = y_train_tensor
            #y_test = y_test_tensor


        pipeline, param_distributions = model_function(config, pre_processing_function_2, running_descriptor, x_test, x_train, y_test, y_train)

        optimization_function = Optimization_algorithms[config.optimization_algorithm]
    
        r2, rmse, y_best_predict, optimizer_object = optimization_function(config, pipeline, param_distributions, x_test, x_train, y_test, y_train)
        
        best_model = optimizer_object.best_estimator_
        
########################################################################################
        ### SAVING PICKLE FILES
        if run_pickle == "yes":
            import pickle
        
            results_dict = {
                    "x_train": x_train,
                    "x_test": x_test,
                    "feature_names": x_train.columns.tolist(),
                    "model_name": model_name,
                    "best_model": best_model
                    }

            with open(f'../new_results/{config.property}/pickle_files/BEST_best_model_{config.descriptor}_{config.model}_{config.optimization_algorithm}.pickle', 'wb') as f:
                pickle.dump(results_dict, f)
       
            print("PICKLE SAVED")


        #with open(f'../new_results/{config.property}/pickle_files/best_model_{config.descriptor}_{config.model}_{config.optimization_algorithm}.pickle', 'wb') as f:
         #   pickle.dump(optimizer_object.best_estimator_, f)
        #with open(f'../new_results/{config.property}/pickle_files/x_train_{config.descriptor}_{config.model}_{config.optimization_algorithm}.pickle', 'wb') as f: 
         #   pickle.dump(x_train, f)
        #with open(f'../new_results/{config.property}/pickle_files/x_test_{config.descriptor}_{config.model}_{config.optimization_algorithm}.pickle', 'wb') as f: 
         #   pickle.dump(x_test, f)
       # with open(f'../new_results/{config.property}/pickle_files/model_name_{config.descriptor}_{config.model}_{config.optimization_algorithm}.pickle', "wb") as f:
        #    pickle.dump(model_name, f)
        
            print("pickle files saved YAYYYYY")
        else: 
            print("pickle set to no. check config if mistake.")
        if run_parquets == "yes":
            best_model = optimizer_object.best_estimator_
        
        #import shap_analysis
        
        #shap_analysis.running_shap(x_train, x_test, model_name, config)
        #print("shap sucessfully ran")
        #print("files saved")
            results = [] 
            for yt, yp in zip(y_test, y_best_predict):
                results.append({
                    "Descriptor": desc,
                    "Model": model_name,
                    "Optimization": config.optimization_algorithm,
                    "RMSE": rmse,
                    "R2": r2,
                    "y_true": yt,
                    "y_pred": yp,
                    **optimizer_object.best_params_ ##idk if this will work
                    })
        else:
            print("parquets set to no.")
if run_parquets == "yes":
## convert to dataframe (to see and for later)
    results_df = pd.DataFrame(results)
    print("The results as dataframe are:")
    print(results_df[["Descriptor", "Model", "Optimization", "RMSE", "R2", "y_true", "y_pred"]])
## save dataframe as its own output file


    filename = "_".join(map(str, config.optimization_algorithm)) + "_".join(map(str, config.model)) + "_" + "_".join(map(str, config.descriptor))
    results_df.to_parquet(f"../new_results/{config.property}/parquet_files/{filename}.parquet", index=False)
    print(f"results saved as parquet to ../new_results/{config.property}/parquet_files/{filename}.parquet")
else: 
    print("parquet set to no. all done.")

#####################################################
##NOTIFICATION EMAIL
#import notify 
#try: 
 #   notify.notify()
#except: 
 #   print("couldn't sent the email."
#else: 
 #   print("email sent.)
