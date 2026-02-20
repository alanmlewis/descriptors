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
#parquet_path = ("../new_results/{config.property}/parquet_files")


def running_shap(backup_config):
    with open(f'../new_results/{backup_config.property}/pickle_files/BEST_best_model_{backup_config.descriptor}_{backup_config.model}_{backup_config.optimization_algorithm}.pickle', 'rb') as f:
        data = pickle.load(f)
    
    x_train = data["x_train"]
    feature_names = data["feature_names"]
    x_test = data["x_test"]
    best_model = data["best_model"]
    model_name = data["model_name"]
    
    print("TYPE X TEST")
    print(type(x_test))
    print("Loaded model and data successfully")
    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))
    
    print(f"Descriptor for the SHAP analysis was: {backup_config.descriptor}")
    background_size = min(50, len(x_train))
    shap_size = min(100, len(x_test))

    background = x_train.sample(background_size, random_state=42)
    print("TYPE BACKGROUND")
    print(type(background))

    ##exit()    
    if model_name in ["XGBoost", "Random Forest"]:
        explainer = shap.Explainer(best_model, background.iloc[:shap_size])
    elif model_name == "Lasso":
        explainer = shap.Explainer(best_model.predict, background.iloc[:shap_size])
    elif model_name == "SVR":
        background_kernel = background.iloc[:min(30, background_size)]
        explainer = shap.KernelExplainer(best_model.predict, background_kernel)
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

        #order = np.argsort(np.abs(shap_values).max(axis=0))[::-1]

        ##shap.plots.beeswarm(
           #         shap.Explanation(
            #                    values=shap_values[:, order],
             #                   data=X_plot.values[:, order],
              #                  feature_names=X_plot.columns[order],
               #                     )
                #    )
    else:
        X_plot = x_test.iloc[:shap_size]
        values = shap_values.values
        base_values = shap_values.base_values

    shap.plots.beeswarm(
        shap.Explanation(
            values=values,
            base_values=base_values,
            data=X_plot,
            feature_names=X_plot.columns.tolist()
            )
        )
    plt.tight_layout()
    plt.title(f"SHAP plot for {backup_config.property} using {backup_config.model} and {backup_config.descriptor}")
    plt.savefig(f"../new_results/{backup_config.property}/SHAP_pickle/shap_{backup_config.descriptor}_{model_name}_{backup_config.optimization_algorithm}.png")
    plt.close()
    print("SHAP ran correctly, and file was saved")
    return shap_values


run_shap = running_shap(backup_config)

