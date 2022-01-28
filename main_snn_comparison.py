#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jan 26 11:24 2022

Auto ML workflow for pre-existing folds

Performs:
    - kNN Imputation on the original data
    - Monte Carlo cross-validation
    - Fold-wise MRMR feature selection
    - Fold-wise SMOTE
    - Auto ML
        - XGBoost
        - SVM

# TODO: activate parameter grid for RF and NN

@author: cspielvogel
"""


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

import pymrmr

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
import metrics
from feature_selection import univariate_feature_selection


def main():
    orig_fold_path = ""

    label_name = ""
    features_to_remove = []

    # Read fold-wise data

    # kNN Imputation on the original data

    # Save imputed data

    # Monte Carlo cross-validation

    # Fold-wise MRMR feature selection

    # Save folds with selected features-only

    # df = pd.read_csv('test_colon_s3.csv')
    # pymrmr.mRMR(df, 'MIQ', 10)

    # Fold-wise SMOTE

    # Save folds with SMOTE applied

    # Classification

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
    dt_param_grid = {"splitter": ["best", "random"],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}

    rf = RandomForestClassifier(n_estimators=100,
                                criterion="entropy",
                                max_depth=5,
                                min_samples_split=5,
                                min_samples_leaf=2)
    # rf_param_grid = {"n_estimators": [100, 500],
    #                  "max_depth": np.arange(1, 20),
    #                  "min_samples_split": [2, 4, 6],
    #                  "min_samples_leaf": [1, 3, 5, 6],
    #                  "max_features": ["auto", "sqrt", "log2"]}
    rf_param_grid = {}

    nn = MLPClassifier(hidden_layer_sizes=(32, 64, 32),
                       activation="relu",
                       # early_stopping=True,
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
    results = pd.DataFrame(index=list(clfs.keys()))

    # Iterate over classifiers
    for clf in clfs:
        pass


if __name__ == "__main__":
    main()
