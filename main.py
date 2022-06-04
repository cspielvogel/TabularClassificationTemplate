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

Input data format specifications:
    - As of now, a file path has to be supplied to the main function as string value for the variable "data_path";
    - The file shall be a CSV file separated by a semicolon (;)
    - The file shall have a header, containing the attribute names and the label name
    - The file shall have an index column containing a unique identifier for each instance

@author: cspielvogel
"""

import os
import pickle
import json
import multiprocessing
from joblib import parallel_backend

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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from exploratory_data_analysis import run_eda
import metrics
from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
from feature_selection import univariate_feature_selection, mrmr_feature_selection
from explainability_tools import plot_importances, plot_shap_features, plot_partial_dependences


def main():
    # Set hyperparameters
    num_folds = 5
    # label_name = "DPD_final"
    label_name = "1"
    # label_name = "OS_histo36"
    verbose = True
    # classifiers_to_run = ["ebm", "dt", "knn", "nn", "rf", "xgb"]
    classifiers_to_run = ["knn"]

    # Set output paths
    # output_path = r"C:\Users\cspielvogel\PycharmProjects\HNSCC"
    output_path = r"./"
    eda_result_path = os.path.join(output_path, r"Results/EDA/")
    explainability_result_path = os.path.join(output_path, r"Results/XAI/")
    model_result_path = os.path.join(output_path, r"Results/Models/")
    performance_result_path = os.path.join(output_path, r"Results/Performance/")
    intermediate_data_path = os.path.join(output_path, r"Results/Intermediate_Data")

    # Specify data location
    # data_path = "/home/cspielvogel/DataStorage/Bone_scintigraphy/Data/umap_feats_pg.csv"
    data_path = r"Data/test_data.csv"
    # data_path = r"C:\Users\cspielvogel\Downloads\fdb_multiomics_w_labels_all.csv"

    # Create save directories if they do not exist yet
    for path in [eda_result_path, explainability_result_path, model_result_path, performance_result_path,
                 intermediate_data_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Load data to table
    df = pd.read_csv(data_path, sep=";", index_col=0)

    # Perform EDA and save results
    run_eda(features=df.drop(label_name, axis="columns"),
            labels=df[label_name],
            label_column=label_name,
            save_path=eda_result_path,
            verbose=verbose)

    # Perform standardized preprocessing
    preprocessor = TabularPreprocessor()
    df = preprocessor.fit_transform(df, label_name=label_name)

    # Separate data into training and test
    y = df[label_name]
    # y = (df[label_name] < 2) * 1    # TODO: remove; only for PG classification
    x = df.drop(label_name, axis="columns")
    feature_names = x.columns

    # Setup classifiers
    ebm = ExplainableBoostingClassifier()
    ebm_param_grid = {"max_bins": [256],
                      "max_interaction_bins": [64],
                      "binning": ["quantile"],
                      "interactions": [15],
                      "outer_bags": [8, 16],
                      "inner_bags": [0, 8],
                      "learning_rate": [0.001, 0.01],
                      "early_stopping_rounds": [50],
                      "early_stopping_tolerance": [0.0001],
                      "max_rounds": [7500],
                      "min_samples_leaf": [2, 4],
                      "max_leaves": [3],
                      "n_jobs": [10]}
    ebm_param_grid = {}     # TODO: reactivate parameter grids

    knn = KNeighborsClassifier()
    knn_param_grid = {"weights": ["distance"],
                      "n_neighbors": [int(val) for val in np.round(np.sqrt(x.shape[1])) + np.arange(5) + 1] +
                                     [int(val) for val in np.round(np.sqrt(x.shape[1])) - np.arange(5) if val >= 1],
                      "p": np.arange(1, 5)}
    knn_param_grid = {}

    dt = DecisionTreeClassifier()
    dt_param_grid = {"splitter": ["best", "random"],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    dt_param_grid = {}

    nn = MLPClassifier()
    nn_param_grid = {"hidden_layer_sizes": [(32, 64, 32)],
                     "early_stopping": [True],
                     "n_iter_no_change": [20],
                     "max_iter": [1000],
                     "activation": ["relu", "tanh", "logistic"],
                     "solver": ["adam"],
                     "learning_rate_init": [0.01, 0.001, 0.0001]}
    nn_param_grid = {}

    rf = RandomForestClassifier()
    rf_param_grid = {"criterion": ["entropy"],
                     "n_estimators": [500],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    rf_param_grid = {}

    xgb = XGBClassifier()
    xgb_param_grid = {"learning_rate": [0.20, 0.30],
                      "max_depth": [4, 6, 8],
                      "min_child_weight": [1, 3],
                      "gamma": [0.0, 0.2],
                      "colsample_bytree": [0.5, 0.7, 1.0]}
    # xgb_param_grid = {}

    clfs = {"ebm":
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

    clfs_performance = {"acc": [], "sns": [], "spc": [], "auc": []}

    # Get number of classes
    num_classes = len(np.unique(y))

    # Initialize result table
    results = pd.DataFrame(index=classifiers_to_run)

    # Iterate over classifiers
    clf_index = -1
    for clf in clfs:
        if clf not in classifiers_to_run:
            continue

        clf_index += 1  # Update index for classifiers which are actually run

        if verbose is True:
            print(f"[Model training] Starting training for {clf} classifier")

        # Initialize cumulated confusion matrix and fold-wise performance containers
        cms = np.zeros((num_classes, num_classes))
        performance_foldwise = {"acc": [], "sns": [], "spc": [], "auc": []}

        # Iterate over MCCV
        tqdm_bar = tqdm(np.arange(num_folds))
        for fold_index in tqdm_bar:

            # Split into training and test data
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.15,
                                                                stratify=y,
                                                                shuffle=True,
                                                                random_state=fold_index)

            # Perform standardization and feature imputation
            intra_fold_preprocessor = TabularIntraFoldPreprocessor(imputation_method="mice",
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

            # Setup model
            model = clfs[clf]["classifier"]
            model.random_state = fold_index

            # Hyperparameter tuning and keep model trained with the best set of hyperparameters
            optimized_model = RandomizedSearchCV(model,
                                                 param_distributions=clfs[clf]["parameters"],
                                                 cv=5,
                                                 random_state=fold_index)
            optimized_model.fit(x_train, y_train)

            # Compute and display fold-wise performance
            y_pred = optimized_model.best_estimator_.predict(x_test)

            # Compute permutation feature importance scores on training and validation data
            raw_importances_train = permutation_importance(optimized_model.best_estimator_, x_test, y_test,
                                                           n_repeats=30,
                                                           scoring="roc_auc_ovr",
                                                           n_jobs=10,  # TODO: adjust and automate
                                                           random_state=fold_index)
            raw_importances_val = permutation_importance(optimized_model.best_estimator_, x_train, y_train,
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
            auc = metrics.roc_auc(y_test, y_pred)

            # Append performance to fold-wise and overall containers
            cms += cm
            performance_foldwise["acc"].append(acc)
            performance_foldwise["sns"].append(sns)
            performance_foldwise["spc"].append(spc)
            performance_foldwise["auc"].append(auc)

            # Progressbar
            if verbose is True:
                tqdm_bar.set_description(str(f"[Model training] Finished classifier {clf_index+1} /"
                                             f" {len(classifiers_to_run)} ({clf}) | Fold {fold_index+1} / {num_folds}"))

        # Setup final model
        seed = 0
        model = clfs[clf]["classifier"]
        model.random_state = seed

        # Preprocess data for creation of final model
        intra_fold_preprocessor = TabularIntraFoldPreprocessor(k="automated", normalization="standardize")
        x_preprocessed = intra_fold_preprocessor.fit_transform(x)

        # Save preprocessed data
        x_preprocessed_df = pd.DataFrame(data=x_preprocessed,
                                         index=x.index,
                                         columns=feature_names)
        x_preprocessed_df.to_csv(os.path.join(output_path,
                                              os.path.join(intermediate_data_path, "preprocessed_features.csv")),
                                 sep=";")

        y_df = pd.DataFrame(data=y, columns=[label_name])
        y_df.to_csv(os.path.join(output_path,
                                 os.path.join(intermediate_data_path, "preprocessed_labels.csv")),
                    sep=";")

        # Feature selection for final model
        selected_indices_preprocessed, x_preprocessed, _ = mrmr_feature_selection(x_preprocessed,
                                                                                  y,
                                                                                  # score_func=f_classif,
                                                                                  num_features="log2n")

        feature_names_selected = feature_names[selected_indices_preprocessed]

        # Hyperparameter tuning for final model
        optimized_model = RandomizedSearchCV(model,
                                             param_distributions=clfs[clf]["parameters"],
                                             cv=10,
                                             random_state=seed)
        optimized_model.fit(x_preprocessed, y)

        # Add feature names enabling interpretability of output plots (only needed for some algorithms like ebm)
        try:
            model.set_params(**{"feature_names": feature_names_selected})
        except ValueError:
            pass

        # Save final model to file
        with open(os.path.join(model_result_path, f"{clf}_model.pickle"), "wb") as file:
            pickle.dump(optimized_model.best_estimator_, file)

        # Save hyperparameters of final model to JSON file
        with open(os.path.join(model_result_path, f"{clf}_optimized_hyperparameters.json"), "w") as file:
            param_dict_json_conform = {}     # Necessary since JSON doesn't take some data types such as int32
            for key in optimized_model.best_params_:
                try:
                    param_dict_json_conform[key] = float(optimized_model.best_params_[key])
                except ValueError:
                    param_dict_json_conform[key] = optimized_model.best_params_[key]

            json.dump(param_dict_json_conform, file, indent=4)

        # Partial dependenced plots (DPD)
        plot_partial_dependences(model=optimized_model,
                                 x=x_preprocessed,
                                 y=y,
                                 feature_names=feature_names_selected,
                                 clf_name=clf,
                                 save_path=explainability_result_path)

        # SHAP analysis and plotting
        plot_shap_features(model=optimized_model,
                           x=x_preprocessed,
                           num_classes=num_classes,
                           feature_names=feature_names_selected,
                           index_names=x_preprocessed_df.index,
                           clf_name=clf,
                           save_path=explainability_result_path,
                           verbose=True)

        # Get mean of feature importance scores and standard deviation over all folds
        overall_mean_importances_train = raw_importances_foldwise_mean_train / num_folds
        overall_std_importances_train = raw_importances_foldwise_std_train / num_folds
        overall_mean_importances_val = raw_importances_foldwise_mean_val / num_folds
        overall_std_importances_val = raw_importances_foldwise_std_val / num_folds

        # Plot feature importances as determined using training and validation data
        plot_title_permutation_importance = f"permutation_importance_{clf}"
        plot_importances(importances_mean=overall_mean_importances_train,
                         importances_std=overall_std_importances_train,
                         feature_names=feature_names_selected,
                         plot_title=plot_title_permutation_importance + "-training_data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=False,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-train"))

        plot_importances(importances_mean=overall_mean_importances_val,
                         importances_std=overall_std_importances_val,
                         feature_names=feature_names_selected,
                         plot_title=plot_title_permutation_importance + "-validation_data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=False,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-test"))

        # Calculate overall performance
        for metric in performance_foldwise:
            avg_metric = np.round(np.sum(performance_foldwise[metric]) / len(performance_foldwise[metric]), 2)
            clfs_performance[metric].append(avg_metric)

        if verbose is True:
            print(str(f"[Model validation] Finished {clf} with MCCV-wide AUC [{clfs_performance['auc'][clf_index]}] and"
                      f" confusion matrix:"))
            print(cms)

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

    # Append performance to result table
    for metric in clfs_performance:
        results[metric] = clfs_performance[metric]

    # Save result table
    colors = ["dimgray", "gray", "darkgray", "maroon", "lightgray", "gainsboro"]
    results.to_csv(os.path.join(performance_result_path, "performances.csv"), sep=";")
    results.plot.bar(rot=45, color=colors).legend(loc="upper right")

    if verbose is True:
        print("[Results] Displaying performance")
        print(results)

    # Adjust legend position so it doesn't mask any bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, clfs_performance.keys(), loc="best", bbox_to_anchor=(1.13, 1.15))

    # Save and display performance plot
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(linestyle="dashed", axis="y")
    plt.title("Overall performance")
    plt.savefig(os.path.join(performance_result_path, "performance.png".format(clf)))
    plt.close()


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    allocated_threads = cpu_count - 2

    with parallel_backend("threading", n_jobs=allocated_threads):
        main()
