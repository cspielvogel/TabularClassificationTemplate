#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:56 2021

Template for binary classifications of tabular data including preprocessing
# TODO: Implement OO version of main function contents
# TODO: Add output of optimal parameters (?)
# TODO: Add permutation feature importance measurements + visualizations
# TODO: Add one way partial dependence plots and maybe two way for top features
# TODO: Add confusion matrix plot / table to output
# TODO: Add analysis flow diagram to README
# TODO: Add commandline options /w argparse
# TODO: Add option for output path

# Optional:
    # TODO: Add local feature importance measurements e.g. SHAP features
    # TODO: Add exploratory analysis with pandas profiling
    # TODO: Add surrogate models to identify potential Rashomon effect

Content:
    - Feature selection
    - Training and evaluation of multiple classification algorithms
        - k-nearest neighbors
        - Decision tree
        - Random forest
        - Neural network
    - Visualization of performance evaluation
        - Barplot with performances for the classification models

Input data format specifications:
    - As of now, a file path has to be supplied to the main function as string value for the variable "data_path";
    - The file shall be a CSV file separated by a semicolon (;)
    - The file shall have a header, containing the attribute names and the label name
    - The file shall have an index column containing a unique identifier for each instance

@author: cspielvogel
"""

import sys
import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from pandas_profiling import ProfileReport

import metrics
from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
from feature_selection import univariate_feature_selection, mrmr_feature_selection
from explainability_tools import plot_importances


def main():
    # Set hyperparameters
    num_folds = 5
    # label_name = "DPD_final"
    label_name = "1"
    features_to_remove = []
    verbose = True
    classifiers_to_run = ["dt", "knn", "rf", "nn"]
    explainability_result_path = "Results/XAI/"
    model_result_path = "Results/Models/"
    performance_save_path = "Results/Performance/"

    # Specify data location
    # data_path = "/home/cspielvogel/ImageAssociationAnalysisKeggPathwayGroups/Data/Dedicaid/fdb_multiomics_w_labels_bonferroni_significant_publication_OS36.csv"
    # data_path = "/home/cspielvogel/DataStorage/Bone_scintigraphy/Data/umap_feats_pg.csv"
    data_path = r"C:\Users\cspielvogel\PycharmProjects\TabularClassificationTemplate\Data\test_data.csv"

    # Load data to table
    df = pd.read_csv(data_path, sep=";", index_col=0)

    # Remove features to be removed
    for col in features_to_remove:
        df = df.drop(col, axis="columns")

    # Only keep first instance if multiple instances have the same key
    num_instances_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    num_instances_diff = num_instances_before - len(df)
    if num_instances_diff > 0:
        print("Warning: {} instances removed due to duplicate keys - only keeping first occurrence!"
              .format(num_instances_diff))

    # Perform standardized preprocessing
    preprocessor = TabularPreprocessor()
    df = preprocessor.fit_transform(df)

    # Separate data into training and test
    y = df[label_name]
    # y = (df[label_name] < 2) * 1    # TODO: remove; only for PG classification
    x = df.drop(label_name, axis="columns")
    feature_names = x.columns

    # Get number of classes
    num_classes = len(np.unique(y))

    # Create and save EDA via Pandas Profiling
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    os.makedirs("Results/EDA/", exist_ok=True)
    profile.to_file("Results/EDA/exploratory_data_analysis.html")

    # Setup classifiers
    knn = KNeighborsClassifier(weights="distance")
    knn_param_grid = {"n_neighbors": [int(val) for val in np.round(np.sqrt(x.shape[1])) + np.arange(5) + 1] +
                                     [int(val) for val in np.round(np.sqrt(x.shape[1])) - np.arange(5) if val >= 1],
                      "p": np.arange(1, 5)}
    knn_param_grid = {}

    dt = DecisionTreeClassifier()
    dt_param_grid = {"splitter": ["best", "random"],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    dt_param_grid = {}  # TODO: reactivate parameter grids

    rf = RandomForestClassifier(n_estimators=100,
                                criterion="entropy",
                                max_depth=5,
                                min_samples_split=5,
                                min_samples_leaf=2)
    rf_param_grid = {"n_estimators": [100, 500],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    rf_param_grid = {}

    nn = MLPClassifier(hidden_layer_sizes=(32, 64, 32),
                       activation="relu",
                       early_stopping=True,
                       n_iter_no_change=20,
                       max_iter=1000)

    # nn_param_grid = {"activation": ["relu", "tanh", "logistic"],
    #                  "solver": ["adam"],
    #                  "learning_rate_init": [0.01, 0.001, 0.0001]}
    nn_param_grid = {}

    clfs = {"knn":
                {"classifier": knn,
                 "parameters": knn_param_grid},
            "dt":
                {"classifier": dt,
                 "parameters": dt_param_grid},
            "rf":
                {"classifier": rf,
                 "parameters": rf_param_grid},
            "nn":
                {"classifier": nn,
                 "parameters": nn_param_grid}}

    clfs_performance = {"acc": [], "sns": [], "spc": [], "auc": []}

    # Initialize result table
    results = pd.DataFrame(index=classifiers_to_run)    # index=list(clfs.keys()))

    # Iterate over classifiers
    for clf in clfs:
        if clf not in classifiers_to_run:
            continue

        if verbose is True:
            print(f"Starting training for {clf}..")

        # Initialize cumulated confusion matrix and fold-wise performance containers
        cms = np.zeros((num_classes, num_classes))
        performance_foldwise = {"acc": [], "sns": [], "spc": [], "auc": []}

        # Iterate over MCCV
        for fold_index in np.arange(num_folds):

            # Split into training and test data
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.15,
                                                                stratify=y,
                                                                shuffle=True,
                                                                random_state=fold_index)

            # Perform standardization and feature imputation
            intra_fold_preprocessor = TabularIntraFoldPreprocessor(k="automated", normalization="standardize")
            intra_fold_preprocessor = intra_fold_preprocessor.fit(x_train)
            x_train = intra_fold_preprocessor.transform(x_train)
            x_test = intra_fold_preprocessor.transform(x_test)

            # Perform (ANOVA) feature selection
            selected_indices, x_train, x_test = mrmr_feature_selection(x_train.values,
                                                                       y_train.values,
                                                                       x_test.values,
                                                                       # score_func=f_classif,
                                                                       num_features="log2n")

            # # Random undersampling
            # rus = RandomUnderSampler(random_state=fold_index, sampling_strategy=0.3)
            # x_train, y_train = rus.fit_resample(x_train, y_train)
            #
            # # SMOTE
            # smote = SMOTE(random_state=fold_index, sampling_strategy=1)
            # x_train, y_train = smote.fit_resample(x_train, y_train)

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
            if len(np.unique(y_train)) > 2:
                y_pred = optimized_model.predict_proba(x_test)
            else:
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
                feature_names = feature_names[selected_indices]
                raw_importances_foldwise_mean_train = np.zeros((len(feature_names),))
                raw_importances_foldwise_std_train = np.zeros((len(feature_names),))
                raw_importances_foldwise_mean_val = np.zeros((len(feature_names),))
                raw_importances_foldwise_std_val = np.zeros((len(feature_names),))

            # Add importance scores to overall container
            raw_importances_foldwise_mean_train += raw_importances_train.importances_mean
            raw_importances_foldwise_std_train += raw_importances_train.importances_std
            raw_importances_foldwise_mean_val += raw_importances_val.importances_mean
            raw_importances_foldwise_std_val += raw_importances_val.importances_std

            # # Predict test data using trained model   # TODO: check if this works for multi class classif
            # y_pred = optimized_model.predict(x_test)

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

        # Setup final model
        seed = 0
        model = clfs[clf]["classifier"]
        model.random_state = seed

        # Hyperparameter tuning for final model
        optimized_model = RandomizedSearchCV(model,
                                             param_distributions=clfs[clf]["parameters"],
                                             cv=10,
                                             random_state=seed)
        optimized_model.fit(x_train, y_train)

        # Save final model to file
        if not os.path.exists(model_result_path):
            os.makedirs(model_result_path)

        with open(os.path.join(model_result_path, f"{clf}_model.pickle"), "wb") as file:
            pickle.dump(optimized_model, file)

        # Get mean of feature importance scores and standard deviation over all folds
        overall_mean_importances_train = raw_importances_foldwise_mean_train / num_folds
        overall_std_importances_train = raw_importances_foldwise_std_train / num_folds
        overall_mean_importances_val = raw_importances_foldwise_mean_val / num_folds
        overall_std_importances_val = raw_importances_foldwise_std_val / num_folds

        # Create XAI result folder if doesn't exist
        if not os.path.exists(explainability_result_path):
            os.makedirs(explainability_result_path)

        # Plot feature importances as determined using training and validation data
        plot_title_permutation_importance = f"Permutation importance {clf} "
        plot_importances(importances_mean=overall_mean_importances_train,
                         importances_std=overall_std_importances_train,
                         feature_names=feature_names,
                         plot_title=plot_title_permutation_importance + " - Training data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=True,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-train")
                         )

        plot_importances(importances_mean=overall_mean_importances_val,
                         importances_std=overall_std_importances_val,
                         feature_names=feature_names,
                         plot_title=plot_title_permutation_importance + " - Validation data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=True,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-test")
                         )

        # Calculate overall performance
        for metric in performance_foldwise:
            avg_metric = np.round(np.sum(performance_foldwise[metric]) / len(performance_foldwise[metric]), 2)
            clfs_performance[metric].append(avg_metric)

        if verbose is True:
            print(f"Finished {clf} with MCCV-wide confusion matrix:")
            print(cms)

        # Display confusion matrix
        # sns.heatmap(cms, annot=True, cmap="Blues", fmt="g")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("{} - Confusion matrix".format(clf))
        # plt.savefig("Results/confusion_matrix-{}.png".format(clf))
        # plt.close()

    # Append performance to result table
    for metric in clfs_performance:
        results[metric] = clfs_performance[metric]

    # Save result table
    if not os.path.exists(performance_save_path):
        os.makedirs(performance_save_path)

    colors = ["dimgray", "gray", "darkgray", "maroon", "lightgray", "gainsboro"]
    results.to_csv(os.path.join(performance_save_path, "performances.csv"), sep=";")
    results.plot.bar(rot=45, color=colors).legend(loc="upper right")

    print(results)

    # Adjust legend position so it doesn't mask any bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, clfs_performance.keys(), loc="best", bbox_to_anchor=(1.13, 1.15))

    # Save and display performance plot
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(linestyle="dashed", axis="y")
    plt.title("Overall performance")
    plt.savefig(os.path.join(performance_save_path, "performance.png".format(clf)))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
