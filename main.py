#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:56 2021

Template for binary classifications of tabular data including preprocessing
# TODO: Implement OO version of main function contents
# TODO: Add analysis flow diagram to README
# TODO: Add commandline options /w argparse
# TODO: ensure that variable names used in savefig() don't contain special characters which can't be used in file name
# TODO: Add surrogate models (maybe in a second line analysis script)
# TODO: Add LCE? https://towardsdatascience.com/random-forest-or-xgboost-it-is-time-to-explore-lce-2fed913eafb8
# TODO: Validate functionality of pickled files after loading
# TODO: Silence dtreeviz "findfont" warning
# TODO: SHAP speedup (shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples)
# TODO: Handle NA imputation for categorical values
# TODO: Perform imputation for entire dataset
# TODO: Show feature correlation matrix for entire and final (after mrmr) data set
# TODO: Handle EDA visualizations (scaling!) for cases where there are categorical and/or missing values
# TODO: Add confidence intervals and p values to performance bar plot (Requires reuse of same fold for all classifiers)
# TODO: Save MCCV folds after creation and load for next classifier training
# TODO: Provide option to load custom folds
# TODO: Runtime optimization by parallelizing folds
# TODO: Add quickload function to load all intermediate data and models for custom analysis
# TODO: Replace PDP plots with https://github.com/SauceCat/PDPbox
# TODO: Optional: Add EBM interpretability plots to XAI
# TODO: Add EBM surrogate interpretability plots to XAI
# TODO: Check how to make EBM work with calibration
# TODO: Check why some classifiers don't have the same number of measurements for the original models calibration curve
# TODO: Add Brier scores to output for calibration
# TODO: Add relevant output to log file in results
# TODO: Add all possible classifier hyperparameters to settings.ini file
# TODO: Add SVM classifier
# TODO: Add options in settings to allow to switch on/off components such as individual XAI features, PCA, tSNE, etc

Input data format specifications:
    - As of now, a file path has to be supplied to the main function as string value for the variable "input_data_path";
    - The file shall be a CSV file separated by a semicolon (;)
    - The file shall have a header, containing the attribute names and the label name
    - The file shall have an index column containing a unique identifier for each instance

@author: cspielvogel
"""

import os
import pickle
import json
import configparser

import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from dtreeviz.trees import dtreeviz

import calibration
from exploratory_data_analysis import run_eda
import metrics
from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
from feature_selection import univariate_feature_selection, mrmr_feature_selection
from explainability_tools import plot_importances, plot_shap_features, plot_partial_dependences, surrogate_model


def create_path_if_not_exist(path):
    """
    Create provided path if path does not exist yet

    :param path: String indicating path to check
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def convert_str_to_container(string, container_type="tuple"):
    """
    Take string flanked with brackets and convert to container; contained elements will be auto cast;
    output will be wrapped in a list

    :param string: string which shall be converted
    :param container_type: string indicating which container to use; options are "tuple" and "list"
    :return: converted container object
    """

    if ", " not in string:
        raise ValueError(f"{string} is not a list")

    if container_type == "tuple":
        brackets = ["(", ")"]
    else:
        brackets = ["[", "]"]

    containers_to_be = string.split(brackets[1] + ", ")
    containers_to_be = [val.replace(brackets[0], "").replace(brackets[1], "").split(", ") for val in containers_to_be]

    container_objs = []
    for container_to_be in containers_to_be:
        if container_type == "tuple":
            container_objs.append(tuple([estimate_type(val) for val in container_to_be]))
        else:
            container_objs.append([estimate_type(val) for val in container_to_be])

    return container_objs


def str_to_bool(s):
    """Convert string to boolean"""

    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    raise ValueError("Not Boolean Value!")


def str_to_none(s):
    """Convert string to None"""

    if s == "None":
        return None
    raise ValueError("Not None Value!")


def estimate_type(var):
    """guesses the str representation of the variable"s type"""

    # Return if not string in the first place
    if not isinstance(var, str):
        return var

    # Guess string representation, defaults to string if others don't pass
    for caster in (str_to_bool, int, float, str_to_none, convert_str_to_container, str):
        try:
            return caster(var)
        except ValueError:
            pass


def auto_cast_string(string):
    """
    Automatic casting of elements within a container

    CAVE: If string to be cast contains a container such as a list, this type must be homogeneous if multiple values
    are included

    :param string: String to be cast
    :return: List containing the casted value(s)
    """

    casted = []

    if string[0] == "(" and string[-1] == ")":
        casted = convert_str_to_container(string, container_type="tuple")
    elif string[0] == "[" and string[-1] == "]":
        casted = convert_str_to_container(string, container_type="list")
    else:
        list_obj = string.split(", ")
        for value in list_obj:
            casted.append(estimate_type(value))

    return casted


def load_hyperparameters(model, config):
    """
    Obtain parameter grids (e.g. for hyperparameter optimization) from loaded configuration file

    :param model: Classifier object implementing .get_params() method such as of type sklearn.base.BaseEstimator
    :param config: configparser.ConfigParser() containing the parameters
    :return: dict containing the parameter grids
    """

    # Initialize output parameter grid
    param_grid = {}

    # Obtain all parameters available for the provided model
    params = model.get_params()

    # Iterate over each parameter
    for param in params:

        try:
            # Cast each string parameter from config file to the most reasonable type
            param_grid[param] = auto_cast_string(config[param])
        except KeyError:
            # Set parameter to default if missing in config file
            param_grid[param] = [model.get_params()[param]]

    return param_grid


def main():
    # Get parameters from settings.ini file
    settings_path = "settings.ini"

    config = configparser.ConfigParser()
    config.read(settings_path)

    verbose = config["Display"]["verbose"] == "True"
    input_data_path = config["Data"]["input_data_file_path"]
    input_file_separator = config["Data"]["input_data_file_separator"]
    output_path = config["Data"]["output_data_folder_path"]
    label_name = config["Data"]["label_column_name"]
    perform_eda = config["EDA"]["perform_eda"] == "True"
    perform_calibration = config["Calibration"]["perform_calibration"] == "True"
    num_folds = int(config["Training"]["number_of_folds"])
    classifiers_to_run = config["Classifiers"]["classifiers_to_run"].split(", ")

    # Set output paths
    eda_result_path = os.path.join(output_path, r"Results/EDA/")
    explainability_result_path = os.path.join(output_path, r"Results/XAI/")
    model_result_path = os.path.join(output_path, r"Results/Models/")
    performance_result_path = os.path.join(output_path, r"Results/Performance/")
    intermediate_data_path = os.path.join(output_path, r"Results/Intermediate_data")
    calibration_path = os.path.join(output_path, r"Results/Calibration")

    # Create save directories if they do not exist yet
    for path in [eda_result_path, explainability_result_path, model_result_path, performance_result_path,
                 intermediate_data_path, calibration_path]:
        create_path_if_not_exist(path)

    # Load data to table
    df = pd.read_csv(input_data_path, sep=input_file_separator, index_col=0)

    if perform_eda is True:
        # Perform EDA and save results
        run_eda(features=df.drop(label_name, axis="columns"),
                labels=df[label_name],
                label_column=label_name,
                save_path=eda_result_path,
                analyses_to_run=["pandas_profiling"],
                verbose=verbose)

    # Perform one hot encoding of categorical features before standard scaling in EDA visualizations
    categorical_mask = df.dtypes == object
    categorical_columns = df.columns[categorical_mask].tolist()

    # Perform standardized preprocessing
    preprocessor = TabularPreprocessor(label_name=label_name,
                                       one_hot_encoder_path=os.path.join(intermediate_data_path,
                                                                         f"one_hot_encoder.pickle"),
                                       label_encoder_path=os.path.join(intermediate_data_path,
                                                                       "label_encoder.pickle"))
    df = preprocessor.fit_transform(df)

    if perform_eda is True:
        # Perform dimensionality reductions
        run_eda(features=df.drop(label_name, axis="columns"),
                labels=df[label_name],
                label_column=label_name,
                save_path=eda_result_path,
                analyses_to_run=["umap", "tsne", "pca"],
                verbose=verbose)

    # Separate data into training and test
    y = df[label_name]
    x = df.drop(label_name, axis="columns")

    feature_names = x.columns

    # Setup classifiers and parameters grids
    ebm = ExplainableBoostingClassifier()
    ebm_hyperparams = config["EBM_hyperparameters"]
    ebm_param_grid = load_hyperparameters(ebm, ebm_hyperparams)
    # ebm_param_grid = {}     # TODO: reactivate parameter grids

    knn = KNeighborsClassifier()
    knn_hyperparams = config["KNN_hyperparameters"]
    knn_param_grid = load_hyperparameters(knn, knn_hyperparams)

    # Set n_neighbors KNN parameter
    if knn_param_grid["n_neighbors"] == ["adaptive"]:
        knn_param_grid["n_neighbors"] = [int(val) for val in np.round(np.sqrt(x.shape[1])) + np.arange(5) + 1] +\
                                        [int(val) for val in np.round(np.sqrt(x.shape[1])) - np.arange(5) if val >= 1]
    # knn_param_grid = {}

    dt = DecisionTreeClassifier()
    dt_hyperparams = config["DT_hyperparameters"]
    dt_param_grid = load_hyperparameters(dt, dt_hyperparams)
    # dt_param_grid = {}

    nn = MLPClassifier()
    nn_hyperparams = config["NN_hyperparameters"]
    nn_param_grid = load_hyperparameters(nn, nn_hyperparams)
    # nn_param_grid = {}

    rf = RandomForestClassifier()
    rf_hyperparams = config["RF_hyperparameters"]
    rf_param_grid = load_hyperparameters(rf, rf_hyperparams)
    # rf_param_grid = {}

    xgb = XGBClassifier()
    xgb_hyperparams = config["XGB_hyperparameters"]
    xgb_param_grid = load_hyperparameters(xgb, xgb_hyperparams)
    # xgb_param_grid = {}

    # Define available classifiers
    available_clfs = {"ebm":
                {"classifier": ebm,
                 "parameters": ebm_param_grid},
            "knn":
                {"classifier": knn,
                 "parameters": knn_param_grid},
            "dt":
                {"classifier": dt,
                 "parameters": dt_param_grid},
            "nn":
                {"classifier": nn,
                 "parameters": nn_param_grid},
            "rf":
                {"classifier": rf,
                 "parameters": rf_param_grid},
            "xgb":
                {"classifier": xgb,
                 "parameters": xgb_param_grid}}

    # Add classifiers to run and initialize performance container holding each classifier performance for each fold fold
    clfs = {}
    performance_clfwise_foldwise = {}
    for clf in classifiers_to_run:
        clfs[clf] = available_clfs[clf]
        performance_clfwise_foldwise[clf] = {"acc": [], "sns": [], "spc": [], "ppv": [], "npv": [], "bacc": [],
                                             "auc": []}

    # Get number of classes
    num_classes = len(np.unique(y))

    # Initialize result table
    results = pd.DataFrame(index=classifiers_to_run)

    if verbose is True:
        print(f"[Model training] Starting model training")

    # Iterate over (MCCV) folds
    tqdm_bar = tqdm(np.arange(num_folds))
    for fold_index in tqdm_bar:

        # Split into training and test data
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.15,
                                                            stratify=y,
                                                            shuffle=True,
                                                            random_state=fold_index)

        # Perform standardization and feature imputation
        intra_fold_preprocessor = TabularIntraFoldPreprocessor(imputation_method="knn",
                                                               k="automated",
                                                               normalization="standardize")
        intra_fold_preprocessor = intra_fold_preprocessor.fit(x_train)
        x_train = intra_fold_preprocessor.transform(x_train)
        x_test = intra_fold_preprocessor.transform(x_test)

        # Perform feature selection
        selected_indices, x_train, x_test = mrmr_feature_selection(x_train.values,
                                                                   y_train.values,
                                                                   x_test.values,
                                                                   # score_func=f_classif,
                                                                   num_features="log2n")
        feature_names_selected = feature_names[selected_indices]

        # SMOTE
        if num_classes == 2:
            smote = SMOTE(random_state=fold_index, sampling_strategy=1)
        else:
            smote = SMOTE(random_state=fold_index, sampling_strategy="not majority")
        x_train, y_train = smote.fit_resample(x_train, y_train)

        # Iterate over classifiers
        for clf in clfs:

            # Initialize cumulated confusion matrix and fold-wise performance containers
            cms = np.zeros((num_classes, num_classes))

            # Setup model
            model = clfs[clf]["classifier"]
            model.random_state = fold_index

            # Hyperparameter tuning and keep model trained with the best set of hyperparameters
            optimized_model = RandomizedSearchCV(model,
                                                 param_distributions=clfs[clf]["parameters"],
                                                 cv=auto_cast_string(config["Training"]["randomizedsearchcv_cv"])[0],
                                                 n_iter=auto_cast_string(config["Training"]["randomizedsearchcv_n_iter"])[0],
                                                 scoring="roc_auc",
                                                 random_state=fold_index)
            optimized_model.fit(x_train, y_train)

            # Compute and display fold-wise performance
            y_pred = optimized_model.predict(x_test)

            # Compute permutation feature importance scores on training and validation data
            raw_importances_train = permutation_importance(optimized_model, x_test, y_test,
                                                           n_repeats=30,
                                                           scoring="roc_auc_ovr",
                                                           n_jobs=10,  # TODO: adjust and automate
                                                           random_state=fold_index)
            raw_importances_val = permutation_importance(optimized_model, x_train, y_train,
                                                         n_repeats=30,
                                                         scoring="roc_auc_ovr",
                                                         n_jobs=10,  # TODO: adjust and automate
                                                         random_state=fold_index)

            # Initialize fold-wise feature importance containers for mean and standard deviation with zero values
            if "raw_importances_foldwise_mean_train" not in locals():
                raw_importances_foldwise_mean_train = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_std_train = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_mean_val = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_std_val = np.zeros((len(feature_names_selected),))

            # Add importance scores to overall container
            raw_importances_foldwise_mean_train += raw_importances_train.importances_mean
            raw_importances_foldwise_std_train += raw_importances_train.importances_std
            raw_importances_foldwise_mean_val += raw_importances_val.importances_mean
            raw_importances_foldwise_std_val += raw_importances_val.importances_std

            # Compute performance
            cm = confusion_matrix(y_test, y_pred)
            acc = metrics.accuracy(y_test, y_pred)
            sns = metrics.sensitivity(y_test, y_pred)
            spc = metrics.specificity(y_test, y_pred)
            ppv = metrics.positive_predictive_value(y_test, y_pred)
            npv = metrics.negative_predictive_value(y_test, y_pred)
            bacc = metrics.balanced_accuracy(y_test, y_pred)
            auc = metrics.roc_auc(y_test, y_pred)

            # Append performance to fold-wise and overall containers
            cms += cm
            performance_clfwise_foldwise[clf]["acc"].append(acc)
            performance_clfwise_foldwise[clf]["sns"].append(sns)
            performance_clfwise_foldwise[clf]["spc"].append(spc)
            performance_clfwise_foldwise[clf]["ppv"].append(ppv)
            performance_clfwise_foldwise[clf]["npv"].append(npv)
            performance_clfwise_foldwise[clf]["bacc"].append(bacc)
            performance_clfwise_foldwise[clf]["auc"].append(auc)

            # Progressbar
            if verbose is True:
                tqdm_bar.set_description(str(f"[Model training] Finished fold {fold_index} / {num_folds}"))

    # Initialize overall performance table
    overall_performances = pd.DataFrame(columns=["acc", "sns", "spc", "ppv", "npv", "bacc", "auc"],
                                        index=list(clfs.keys()))

    # Iterate over classifiers and aggregate fold-wise performances
    for clf in clfs:

        # Calculate overall performance
        for metric in performance_clfwise_foldwise[clf]:
            performance_mean = np.mean(performance_clfwise_foldwise[clf][metric])
            performance_mean_rounded = np.round(performance_mean, 2)
            overall_performances.at[clf, metric] = performance_mean_rounded

        # Display and save confusion matrix figure
        seaborn.heatmap(cms, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("{} - Confusion matrix".format(clf))
        plt.savefig(os.path.join(performance_result_path, f"confusion_matrix-{clf}.png"))
        plt.close()

        # Save confusion matrix as CSV
        label_names = np.sort(np.unique(y))
        cm_df = pd.DataFrame(cms,
                             columns=[str(name) + " (actual)" for name in label_names],
                             index=[str(name) + " (predicted)" for name in label_names])
        cm_df.to_csv(os.path.join(performance_result_path, f"confusion_matrix-{clf}.csv"), sep=";")

    # Save result table with all classifiers performances
    colors = ["dimgray", "gray", "darkgray", "lightgray", "gainsboro", "whitesmoke", "maroon"]
    overall_performances.to_csv(os.path.join(performance_result_path, "performances.csv"), sep=";")
    overall_performances.plot.bar(rot=45, color=colors).legend(loc="upper right")

    if verbose is True:
        print("[Results] Displaying performance")
        print(overall_performances)

    # Adjust legend position so it doesn't mask any bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, overall_performances.columns, loc="best", bbox_to_anchor=(1.13, 1.15))

    # Save and display performance plot
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(linestyle="dashed", axis="y")
    plt.title("Overall performance")
    plt.savefig(os.path.join(performance_result_path, "performances.png".format(clf)))
    plt.close()

    # Iterate over classifiers, create final models and apply XAI techniques
    for clf in clfs:

        # Setup final model
        seed = 0
        model = clfs[clf]["classifier"]
        model.random_state = seed

        # Preprocess data for creation of final model
        intra_fold_preprocessor = TabularIntraFoldPreprocessor(k="automated",
                                                               normalization="standardize",
                                                               imputer_path=os.path.join(intermediate_data_path,
                                                                                         f"{clf}_scaler.pickle"),
                                                               scaler_path=os.path.join(intermediate_data_path,
                                                                                        f"{clf}_imputer.pickle"))
        x_preprocessed = intra_fold_preprocessor.fit_transform(x)

        # Save preprocessed data
        x_preprocessed_df = pd.DataFrame(data=x_preprocessed,
                                         index=x.index,
                                         columns=feature_names)
        x_preprocessed_df.to_csv(os.path.join(intermediate_data_path, "preprocessed_features.csv"),
                                 sep=";")

        y_df = pd.DataFrame(data=y, columns=[label_name])
        y_df.to_csv(os.path.join(intermediate_data_path, "preprocessed_labels.csv"), sep=";")

        # Feature selection for final model
        selected_indices_preprocessed, x_preprocessed, _ = mrmr_feature_selection(x_preprocessed,
                                                                                  y,
                                                                                  # score_func=f_classif,
                                                                                  num_features="log2n")

        # Get feature names selected
        feature_names_selected = feature_names[selected_indices_preprocessed]

        # Add feature names enabling interpretability of output plots (only needed for some algorithms like ebm)
        if clf == "ebm":
            model.set_params(**{"feature_names": feature_names_selected})

        # Hyperparameter tuning for final model
        optimized_model = RandomizedSearchCV(model,
                                             param_distributions=clfs[clf]["parameters"],
                                             cv=auto_cast_string(config["Training"]["randomizedsearchcv_cv"])[0],
                                             n_iter=auto_cast_string(config["Training"]["randomizedsearchcv_n_iter"])[0],
                                             scoring="roc_auc",
                                             random_state=seed)
        optimized_model.fit(x_preprocessed, y)
        best_params = optimized_model.best_params_
        optimized_model = optimized_model.best_estimator_

        if perform_calibration is True:

            # Perform probability calibration of final model using ensemble approach
            optimized_model = calibration.calibrate(model=optimized_model,
                                                    features=x_preprocessed,
                                                    labels=y,
                                                    calibration_path=calibration_path,
                                                    clf_name=clf,
                                                    verbose=verbose)

        # Save final model to file
        with open(os.path.join(model_result_path, f"{clf}_model.pickle"), "wb") as file:
            pickle.dump(optimized_model, file)

        # Save hyperparameters of final model to JSON file
        with open(os.path.join(model_result_path, f"{clf}_optimized_hyperparameters.json"), "w") as file:
            param_dict_json_conform = {}     # Necessary since JSON doesn't take some data types such as int32
            for key in best_params:
                try:
                    param_dict_json_conform[key] = float(best_params[key])
                except ValueError:
                    param_dict_json_conform[key] = best_params[key]
                except TypeError:
                    param_dict_json_conform[key] = best_params[key]

            json.dump(param_dict_json_conform, file, indent=4)

        # Get mean of feature importance scores and standard deviation over all folds
        overall_mean_importances_train = raw_importances_foldwise_mean_train / num_folds
        overall_std_importances_train = raw_importances_foldwise_std_train / num_folds
        overall_mean_importances_val = raw_importances_foldwise_mean_val / num_folds
        overall_std_importances_val = raw_importances_foldwise_std_val / num_folds

        # # Plot feature importances as determined using training and validation data
        # plot_title_permutation_importance = f"permutation_importance_{clf}"
        # importances_save_path = os.path.join(explainability_result_path,
        #                                      "Permutation_importances")
        # create_path_if_not_exist(importances_save_path)
        # train_importances_save_path = os.path.join(importances_save_path,
        #                                            plot_title_permutation_importance + "-train")
        # plot_importances(importances_mean=overall_mean_importances_train,
        #                  importances_std=overall_std_importances_train,
        #                  feature_names=feature_names_selected,
        #                  plot_title=plot_title_permutation_importance + "-training_data",
        #                  order_alphanumeric=True,
        #                  include_top=0,
        #                  display_plots=False,
        #                  save_path=train_importances_save_path)
        #
        # test_importances_save_path = os.path.join(importances_save_path,
        #                                           plot_title_permutation_importance + "-test")
        # plot_importances(importances_mean=overall_mean_importances_val,
        #                  importances_std=overall_std_importances_val,
        #                  feature_names=feature_names_selected,
        #                  plot_title=plot_title_permutation_importance + "-validation_data",
        #                  order_alphanumeric=True,
        #                  include_top=0,
        #                  display_plots=False,
        #                  save_path=test_importances_save_path)
        #
        # # Partial dependenced plots (DPD)
        # pdp_save_path = os.path.join(explainability_result_path, "Partial_dependence_plots")
        # create_path_if_not_exist(pdp_save_path)
        # plot_partial_dependences(model=optimized_model,
        #                          x=x_preprocessed,
        #                          y=y,
        #                          feature_names=feature_names_selected,
        #                          clf_name=clf,
        #                          save_path=pdp_save_path)
        #
        # # Create surrogate models
        # if clf not in ["dt", "ebm"]:
        #     surrogate_models_save_path = os.path.join(explainability_result_path, "Surrogate_models")
        #     create_path_if_not_exist(surrogate_models_save_path)
        #
        #     # Decision tree surrogate model
        #     dt_surrogate_models_save_path = os.path.join(surrogate_models_save_path,
        #                                                  f"dt_surrogate_model_for-{clf}.pickle")
        #     dt_surrogate_params = {     # TODO: allow customization
        #         "max_depth": 3,
        #         "min_samples_leaf": 3
        #     }
        #     dt_surrogate_model, _ = surrogate_model(opaque_model=optimized_model,
        #                                             features=x_preprocessed,
        #                                             params=dt_surrogate_params,
        #                                             surrogate_type="dt",
        #                                             save_path=dt_surrogate_models_save_path,
        #                                             verbose=True)
        #
        #     # EBM surrogate model
        #     ebm_surrogate_models_save_path = os.path.join(surrogate_models_save_path,
        #                                                   f"ebm_surrogate_model_for-{clf}.pickle")
        #     ebm_surrogate_params = {}
        #     ebm_surrogate_model, _ = surrogate_model(opaque_model=optimized_model,
        #                                              features=x_preprocessed,
        #                                              params=ebm_surrogate_params,
        #                                              surrogate_type="ebm",
        #                                              save_path=ebm_surrogate_models_save_path,
        #                                              verbose=True)
        #
        #     # Create and save surrogate tree visualization
        #     dt_surrogate_model_visualization_save_path = os.path.join(surrogate_models_save_path,
        #                                                               f"dt_surrogate_model_for-{clf}.svg")
        #     int_label_names = [label for label in dt_surrogate_model.classes_]
        #     if preprocessor.label_encoder != None:
        #         decoded_class_names = list(preprocessor.label_encoder.inverse_transform(int_label_names))
        #     else:
        #         decoded_class_names = int_label_names
        #
        #     viz = dtreeviz(dt_surrogate_model, x_preprocessed, y,
        #                    target_name="Label",
        #                    feature_names=feature_names_selected,
        #                    class_names=decoded_class_names)
        #     viz.save(dt_surrogate_model_visualization_save_path)
        #
        # # SHAP analysis and plotting
        # shap_save_path = os.path.join(explainability_result_path, "SHAP")
        # create_path_if_not_exist(shap_save_path)
        # if preprocessor.label_encoder != None:
        #     decoded_class_names = list(preprocessor.label_encoder.inverse_transform(optimized_model.classes_))
        # else:
        #     decoded_class_names = optimized_model.classes_
        #
        # plot_shap_features(model=optimized_model,
        #                    x=x_preprocessed,
        #                    feature_names=feature_names_selected,
        #                    index_names=x_preprocessed_df.index,
        #                    clf_name=clf,
        #                    classes=decoded_class_names,
        #                    save_path=shap_save_path,
        #                    verbose=True)


if __name__ == "__main__":
    main()
