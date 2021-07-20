#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:56 2021

Template for binary classifications of tabular data including preprocessing
# TODO: Implement OO version of main function contents
# TODO: Ensure reproducibility

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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
import metrics
from feature_selection import univariate_feature_selection


def main():
    # Set hyperparameters
    num_folds = 100
    label_name = "DFS_histo_median"
    features_to_remove = ["DFS_histo24", "DFS_histo36"]

    # Specify data location
    # data_path = "Data/test_data.csv"

    data_path = "/home/cspielvogel/ImageAssociationAnalysisKeggPathwayGroups/Data/Dedicaid/fdb_multiomics_w_labels.csv"

    # Load data to table
    # df = pd.read_csv(data_path, sep=";", index_col=0)
    df = pd.read_csv(data_path, sep=";")
    print(df)
    # Check if any labels are missing
    print("Number of missing values:\n", df.isnull().sum())
    print()

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

    # Display bar chart with number of samples per class
    # seaborn.countplot(x=label_name, data=df)
    # plt.title("Original class frequencies")
    # plt.savefig("Results/original_class_frequencies.png")
    # plt.close()

    # Separate data into training and test
    y = df[label_name]
    x = df.drop(label_name, axis="columns")

    # Get samples per class
    print("Samples per class")
    for (label, count) in zip(*np.unique(y, return_counts=True)):
        print("{}: {}".format(label, count))
    print()

    # Get number of classes
    num_classes = len(np.unique(df[label_name].values))

    # Setup classifiers
    knn = KNeighborsClassifier(weights="distance")
    knn_param_grid = {"n_neighbors": [int(val) for val in np.round(np.sqrt(x.shape[1])) + np.arange(5) + 1] +
                                     [int(val) for val in np.round(np.sqrt(x.shape[1])) - np.arange(5) if val >= 1],
                      "p": np.arange(1, 5)}

    dt = DecisionTreeClassifier()
    dt_param_grid = {"criterion": ["gini", "entropy"],
                     "splitter": ["best", "random"],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}

    rf = RandomForestClassifier(n_estimators=100,
                                criterion="entropy",
                                max_depth=5,
                                min_samples_split=5,
                                min_samples_leaf=2)
    rf_param_grid = {}

    nn = MLPClassifier(hidden_layer_sizes=(32, 64, 32),
                       activation="relu")
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
    results = pd.DataFrame(index=list(clfs.keys()))

    # Iterate over classifiers
    for clf in clfs:

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
            selected_indices, x_train, x_test = univariate_feature_selection(x_train.values,
                                                                             y_train.values,
                                                                             x_test.values,
                                                                             score_func=f_classif,
                                                                             num_features="log2n")

            # # Random undersampling
            # rus = RandomUnderSampler(random_state=fold_index, sampling_strategy=0.3)
            # x_train, y_train = rus.fit_resample(x_train, y_train)

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

            # Predict test data using trained model
            y_pred = optimized_model.predict(x_test)

            # Compute performance
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            sns = metrics.sensitivity(y_test, y_pred)
            spc = metrics.specificity(y_test, y_pred)
            auc = metrics.roc_auc(y_test, y_pred)

            # Append performance to fold-wise and overall containers
            cms += cm
            performance_foldwise["acc"].append(acc)
            performance_foldwise["sns"].append(sns)
            performance_foldwise["spc"].append(spc)
            performance_foldwise["auc"].append(auc)

        # Calculate overall performance
        for metric in performance_foldwise:
            avg_metric = np.round(np.sum(performance_foldwise[metric]) / len(performance_foldwise[metric]), 2)
            clfs_performance[metric].append(avg_metric)

        # Display overall performances
        print("== {} ==".format(clf))
        print("Cumulative CM:\n", cms)
        for metric in clfs_performance:
            print("Avg {}: {}".format(metric, clfs_performance[metric][-1]))
        print()

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
    results.to_csv("performances.csv", sep=";")
    results.plot.bar(rot=45).legend(loc="upper right")
    plt.savefig("performance.png".format(clf))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
