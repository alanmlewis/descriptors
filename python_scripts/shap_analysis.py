## shap analysis
import backup_config
import os
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
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
#parquet_path = ("../new_results/{config.property}/parquet_files")
root_dir = Path(__file__).resolve().parent.parent
#descriptor_underscored = "_".join(backup_config.descriptor[0].split())
pickle_file = root_dir / "new_results"/ f"{backup_config.property[0]}"/"pickle_files"/f"best_model_{backup_config.descriptor[0]}_{backup_config.model[0]}_Random_Search_optimizer.pickle"
RDKit_features_file = root_dir / "python_scripts" / "RDKit_descriptor_features"


def running_shap(backup_config, pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print(type(data))

    
    x_train = data["x_train"]
    if backup_config.descriptor[0] == "RDKit":
        features = pd.read_csv(RDKit_features_file, header=None)[0].tolist()
    x_test = data["x_test"]

    best_model = data["best_model"]
    model_name = data["model_name"]
    print(f"Descriptor for the SHAP analysis was: {backup_config.descriptor[0]}")
    background_size = min(50, len(x_train))
    shap_size = min(100, len(x_test))

    background = x_train.sample(background_size, random_state=42)
    print("TYPE BACKGROUND")
    print(type(background))

    ##exit()    
    if model_name in ["XGBoost", "Random_Forest"]:
        explainer = shap.Explainer(best_model.predict, background.iloc[:shap_size], max_exals=850)
    elif model_name == "Lasso":
        explainer = shap.Explainer(best_model.predict, background.iloc[:shap_size], max_evals=850)
    elif model_name == "SVR":
        background_kernel = background.iloc[:min(30, background_size)]
        explainer = shap.KernelExplainer(best_model.predict, background_kernel, max_exals=850)
    else: 
        raise ValueError(f"Unknown Model, check input. Model inputted was {model_name}")

    if model_name == "SVR":
        shap_values = explainer.shap_values(x_test.iloc[:min(50, shap_size)])
    else: 
        shap_values = explainer(x_test.iloc[:shap_size])
    ##shap.summary_plot(shap_values, x_test_processed, ##all features from the descriptor?? - ouput of running_desciptor idk )
    import numpy as np
    if model_name == "SVR":
        X_plot = x_test.iloc[:min(50, shap_size)]
        values = shap_values
        base_values = None

    else:
        X_plot = x_test.iloc[:shap_size]
        values = shap_values.values
        base_values = shap_values.base_values
    if backup_config.descriptor[0] == "RDKit":
        shap.plots.beeswarm(
            shap.Explanation(
                values=values,
                base_values=base_values,
                data=X_plot,
                feature_names=features
                )
            )
    else:
       shap.plots.beeswarm(
               shap.Explanation(
                   values=values,
                   base_values=base_values,
                   data=X_plot
                   )
               )
    plt.tight_layout()
    plt.title(f"SHAP plot for {backup_config.property[0]} using {backup_config.model[0]} and {backup_config.descriptor[0]}")
    plt.savefig(f"../new_results/{backup_config.property[0]}/SHAP/shap_{backup_config.descriptor[0]}_{model_name}_{backup_config.optimization_algorithm}.png")
    plt.close()
    print("SHAP ran correctly, and file was saved")
    return shap_values


run_shap = running_shap(backup_config, pickle_file)

from rdkit.Chem.Draw import DrawMorganBit

def finding_that_bit(config, bit):
    ##Read SMILES txt directly (one SMILES per line)
    smiles_list = pd.read_csv(config.SMILES_file, header=None, names=["SMILES"])
    smiles_list = smiles_list[['SMILES']]

    # Fingerprint parameter
    fpgen = AChem.GetMorganGenerator(radius=3, fpSize=4096)  # Larger fpSize = fewer collisions

    ##Generate molecules
    mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list['SMILES']]
    bitinfo_all = []
    for m in mol:
        bitInfo = {}
        if m is not None:
            fpgen.GetFingerprint(m, bitInfo=bitInfo)
        
        bitinfo_all.append(bitInfo)
         
    molecules_with_bit = []

    for i, bitinfo in enumerate(bitinfo_all):
        if bit in bitinfo:
            molecules_with_bit.append(i)
    
    print(molecules_with_bit)
    for idx in molecules_with_bit[:5]:
        image = DrawMorganBit(mol[idx], bit, bitinfo_all[idx])
    
        display(image) 
    return

if config.run_bits == "yes":    
    bit_finder = finding_that_bit(config, ##bit num)
