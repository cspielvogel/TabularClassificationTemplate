#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:56 2021

Template for binary classifications of tabular data including preprocessing
# TODO: Implement object oriented version of main function contents

Content:
    - Feature selection
    - TODO: Hyperparameter optimization
    - Training and evaluation of multiple classification algorithms
        - k-nearest neighbors
        - Decision tree
        - Random forest
        - Neural network
    - Visualization of performance evaluation
        - Barplot with performances for the classification models

@author: cspielvogel
"""

import sys

import numpy as np
import pandas as pd

import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import f_classif

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from preprocessing import TabularPreprocessingPipeline
import metrics
from feature_selection import univariate_feature_selection


def main():
    # Set hyperparameters
    num_folds = 100
    # label_name = "Label"
    label_name = "os24_histo"

    # Specify data location
    # data_path = "/home/clemens/Classification_blood71/Data/clinical_wlabels.csv"
    data_path = "/media/cspielvogel/DataStorage/HNSCC/71pat_raw_features/Master_omics_table/FDB_multi-omics_hnscc_wos24.csv"

    # Load data to table
    df = pd.read_csv(data_path, sep=";", index_col=0)

    # Check if any labels are missing
    print("Number of missing values:\n", df.isnull().sum())
    print()

    # Only keep first instance if multiple instances have the same key
    num_instances_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    num_instances_diff = num_instances_before - len(df)
    if num_instances_diff > 0:
        print("Warning: {} instances removed due to duplicate keys - only keeping first occurrence!"
              .format(num_instances_diff))

    # Perform standardized preprocessing
    preprocessor = TabularPreprocessingPipeline()
    df = preprocessor.fit_transform(df)

    # Display bar chart with number of samples per class
    # seaborn.countplot(x=label_name, data=df)
    # plt.title("Original class frequencies")
    # plt.savefig("Results/original_class_frequencies.png")
    # plt.close()

    # Separate data into training and test
    y = df[label_name]
    x = df.drop(label_name, axis="columns")

    # TODO: tmp, remove!
    for col in x.columns:
        if col != "Response" and not (col.startswith("largest") or col.startswith("merged") or col.startswith("multi")):
            x = x.drop(col, axis="columns")

    # Get samples per class
    print("Samples per class")
    for (label, count) in zip(*np.unique(y, return_counts=True)):
        print("{}: {}".format(label, count))
    print()

    # Get number of classes
    num_classes = len(np.unique(df[label_name].values))

    # Setup classifiers
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

    dt = DecisionTreeClassifier(criterion="entropy",
                                max_depth=8,
                                min_samples_leaf=5,
                                min_samples_split=2)

    rf = RandomForestClassifier(n_estimators=100,
                                criterion="entropy",
                                max_depth=5,
                                min_samples_split=5,
                                min_samples_leaf=2)

    nn = MLPClassifier(hidden_layer_sizes=(32, 64, 32),
                       activation="relu")

    clfs = {"kNN": knn, "DT": dt, "RF": rf, "NN": nn}
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
                                                                random_state=fold_index)

            # Perform (ANOVA) feature selection
            selected_indices, x_train, x_test = univariate_feature_selection(x_train.values,
                                                                             y_train.values,
                                                                             x_test.values,
                                                                             score_func=f_classif,
                                                                             num_features="log2n")

            # # Random undersampling
            # rus = RandomUnderSampler(random_state=fold_index, sampling_strategy=0.3)
            # x_train, y_train = rus.fit_resample(x_train, y_train)

            # SMOTE
            smote = SMOTE(random_state=fold_index, sampling_strategy=1)
            x_train, y_train = smote.fit_resample(x_train, y_train)

            # Setup model
            model = clfs[clf]
            model.random_state = fold_index

            # Train model
            model.fit(x_train, y_train)

            # Predict test data using trained model
            y_pred = model.predict(x_test)

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
            print("Avg {}: {}".format(metric, clfs_performance[metric]))
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
    # results.to_csv("Results/performances.csv", sep=";")
    # results.plot.bar(rot=45).legend(loc="upper right")
    # plt.savefig("Results/performance.png".format(clf))
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
