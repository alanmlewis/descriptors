## shap analysis


def running_shap(pipeline, x_train, x_test, y_train, y_test, model_name, config):
    
    import shap 
    import matplotlib.pyplot as plt
    Max_SHAP_samples = 200
    print(f"Descriptor for the SHAP analysis was: {config.descriptor}")
    
    estimator_name = list(pipeline.named_steps.keys())[-1] ## str of model step 
    estimator = pipeline.named_steps[estimator_name] ## getting model -- need model itself for SHAP

    Max_SHAP_samples = 500
    preprocessing_for_NaN = pipeline[:-1] ## all but last step
    x_train_processed = preprocessing_for_NaN.transform(x_train) ##processing data sepatretly for Kernel explainer 
    x_test_processed = preprocessing_for_NaN.transform(x_test)

    if model_name in ["XGBoost", "Random Forest"]:
        explainer = shap.TreeExplainer(estimator, x_train_processed[:Max_SHAP_samples])
    elif model_name = "Lasso":
        explainer = shap.LinearExplainer(estimator, x_train_processed[:Max_SHAP_samples])
    elif model_name = "SVR":
        New_Max_SHAP_samples = 75
        explainer = shap.KernelExplainer(estimator.predict, x_train_processed[:New_Max_SHAP_samples])
    else: 
        raise ValueError(f"Unknown Model, check input. Model inputted was {model_name}")

    if model_name = "SVR":
        shap_values = explainer.shap_values(x_test_processed[:New_Max_SHAP_samples])
    else: 
        shap_values = explainer.shap_values(x_test_processed[:Max_SHAP_samples])
    ##shap.summary_plot(shap_values, x_test_processed, ##all features from the descriptor?? - ouput of running_desciptor idk )
    shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))
    plt.tight_layout()
    plt.savefig(f"../results/{config.property}/SHAP")
    plt.close()
    print("SHAP ran correctly, and file was saved")
    return shap_values

