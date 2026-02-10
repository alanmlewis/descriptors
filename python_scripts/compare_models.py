### this is where i will compare modelsffffff
##didnt work
import backup_config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap
import os
##collecting all the foles with the same descriptor 
from pathlib import Path 
directory = Path(f"../new_results/{backup_config.property}/parquet_files")
parquet_files = list(directory.glob("*.parquet"))
print("current working directory is:", os.getcwd())

################################################
def normalise_all_names(y):
    return y.replace(" ","_").lower()

def get_all_descriptor_files(parquet_files, backup_config):
    target_descriptor = [normalise_all_names(d) for d in backup_config.descriptor]
    ##check im not going totally insane:
    print("All parquet files found are:")
    for p in parquet_files:
        print("   ", p.name)

        print("The target descriptor:", backup_config.descriptor)
    return [
            f for f in parquet_files 
            if any(normalise_all_names(d) in normalise_all_names(f.stem) for d in target_descriptor)
                ] ## if file in paruqet files has the desired descriptor name, add to returned list
        
same_descriptor_parquets = get_all_descriptor_files(parquet_files, backup_config)

##sanity check 2
print("matched parquet files:")
for p in same_descriptor_parquets: 
    print("      ", p.name)
########################################################
##main func for comparing models 
def compare_models(same_descriptor_parquets, backup_config):
    
    dataframes = [pd.read_parquet(p) for p in same_descriptor_parquets]
    dataframe = pd.concat(dataframes, ignore_index = True)
    sns.set(style="whitegrid")
    filename = "_".join(map(str, backup_config.descriptor))
    ## RMSEs comparison
    plt.figure(figsize=(12, 12))
    ax = sns.barplot(data=dataframe, x="Model", y="RMSE", hue="Optimization", palette="Blues")
    #for p in ax.patches: ## for each bar as a rectangle
        #ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha= "center", va="bottom", fontsize = 12) ## add RMSE values at the top of each bar --REMOVED AS MESSING UP AND ITS UNNECESSARY 

    plt.title("RMSE Comparison (lower = better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"../new_results/{backup_config.property}/RMSE_comparison_{filename}.png")
    plt.close()

    ## R2 comparison
    plt.figure(figsize=(12,12))
    for model_name in dataframe["Model"].unique():
        model_data = dataframe[dataframe["Model"] == model_name]
        plt.scatter(model_data["y_true"], model_data["y_pred"], label=f"{model_name} (R2={model_data['R2'].iloc[0]:.2f})", alpha=0.6)## alpha controls transparency of points, set decimal points to 2 atm
    min_val = dataframe["y_true"].min()
    max_val = dataframe["y_true"].max()
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
    model_names = dataframe["Model"].unique()
    model_names_str = " ,".join(model_names)
    wrapped_title = "\n".join(textwrap.wrap(f"R2 comparision - {model_names_str}"))
    plt.title(wrapped_title)
    plt.xlabel("True values") 
    plt.ylabel("Predicted values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../new_results/{backup_config.property}/R2_Plot_{filename}.png")## figure out hwo to add the specific model to this
    plt.close()


    ## pairty + residual
    for (model_name, opt_name), group in dataframe.groupby(["Model", "Optimization"]):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values

        
        spaceless_model_name = model_name.replace(" ", "_") 
        spaceless_opt_name = opt_name.replace(" ", "_")


        ## parity
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        wrapped_title = "\n".join(textwrap.wrap(f"{spaceless_model_name}_{spaceless_opt_name}_True_vs_Predicted"))
        plt.title(wrapped_title)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.tight_layout()
        plt.savefig(f"../new_results/{backup_config.property}/ParityPlot_{filename}_{spaceless_opt_name}")
        plt.close()

        ## residual
        residuals = y_true - y_pred
        plt.figure(figsize=(5, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
        wrapped_title = "\n".join(textwrap.wrap(f"{backup_config.property}_{spaceless_opt_name}_Residuals_vs_Predictions"))
        plt.title(wrapped_title)
        plt.xlabel("Predicted values")
        plt.ylabel("Residual (True values − Predicted values)")
        plt.tight_layout()
        plt.savefig(f"../new_results/{backup_config.property}/ResidualPlot_{filename}_{spaceless_opt_name}")
        plt.close()


if str(backup_config.compare_graphs).lower() == "yes":
    comparing_models = compare_models(same_descriptor_parquets, backup_config)
    print("compare models has run")
else: 
    print("config models comparision set to No.")
##### END OF STUFF TO MOVE#####################


