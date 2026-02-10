## shap analysis
import config
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

#parquet_path = ("../new_results/{config.property}/parquet_files")
def running_shap(best_model, x_train, x_test, model_name, config):
    import shap
    import matplotlib.pyplot as plt
    import os
    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))

    ##Max_SHAP_samples = 200
    print(f"Descriptor for the SHAP analysis was: {config.descriptor}")
    background_size = min(50, len(x_train))
    shap_size = min(100, len(x_test))

    background = x_train.sample(background_size, random_state=42)
    ##Max_SHAP_samples = 500
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
        order = np.argsort(np.abs(shap_values).max(axis=0))[::-1]

        shap.plots.beeswarm(
                    shap.Explanation(
                                values=shap_values[:, order],
                                data=X_plot.values[:, order],
                                feature_names=X_plot.columns[order],
                                    )
                    )
    else:
        shap.plots.beeswarm(shap_values)
    plt.tight_layout()
    plt.title(f"SHAP plot for {config.property} using {model_name} and {config.descriptor}")
    plt.savefig(f"../new_results/{config.property}/SHAP/shap_{config.descriptor}_{model_name}_{config.optimization_algorithm}.png")
    plt.close()
    print("SHAP ran correctly, and file was saved")
    return shap_values


##run_shap = running_shap(best_model, x_train, x_test, model_name, config)

