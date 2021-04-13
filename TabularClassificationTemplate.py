#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mar 15 13:30 2021

Template for binary classifications of tabular data including preprocessing
# TODO: extend to multiclass classifications (metrics!)

Content:
    - Preprocessing pipeline
        - Removing all-NA instances
        - Removing features with too many missing values (>0.2)
        - Filling missing values using kNN imputation
        - TODO: removal of correlated features
    - Performance estimation using Monte Carlo cross validation with multiple metrics
        - Accuracy
        - AUC
        - Sensitivity
        - Specificity
    - TODO: Feature selection
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
import numbers

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class TabularPreprocessingPipeline:
    def __init__(self):
        self.data = None
        self.k = None
        self.max_missing_ratio = None
        self.is_fit = False

    def _remove_partially_missing(self, axis="columns"):
        """
        Removal of features or instances with missing features above the given ratio
        :param axis: string (must be one of "rows" or "columns) indicating whether to remove rows or columns
        :return: pandas.DataFrame without rows or columns with missing values above the max_missing_ratio
        """

        assert axis in ["columns", "rows"], "Parameter 'axis' must be one of ['columns', 'rows']"

        iterable = self.data.columns if axis == "columns" else self.data.index

        for subset in iterable:
            if axis == "columns":
                missing_ratio = self.data.loc[:, subset].isnull().sum() / self.data.shape[0]
            else:
                missing_ratio = self.data.loc[subset, :].isnull().sum() / self.data.shape[1]

            if missing_ratio > self.max_missing_ratio:
                self.data = self.data.drop(subset, axis=axis)

        return self.data

    def fit(self, data, k="standard", max_missing_ratio=0.2):
        """
        Fitting the standard preprocessing pipeline before transformation.
        Includes removal of instances with only missing values, removal of features with more than 20 % missing values,
        kNN-based imputation for missing values, and removal of correlated features.
        :param data: pandas.DataFrame containing the data to be preprocessed
        :param k: int indicating the k nearest neighbors for kNN-based  imputation
        :param max_missing_ratio: float indicating the maximum ratio of missing to all feature values to keep a feature
        :return: TabularPreprocessingPipeline fitted with the given parameters
        """

        # Ensure valid parameters
        assert isinstance(data, pd.DataFrame), "Parameter 'data' must be an instance of pandas.DataFrame"
        assert isinstance(k, numbers.Number) or k == "standard", "Parameter 'k' must either be numeric or 'standard'"
        assert isinstance(max_missing_ratio, numbers.Number), "Parameter 'max_missing_ratio' must be float"
        assert max_missing_ratio <= 1, "Parameter 'max_missing_ratio' must be less or equal 1"
        assert max_missing_ratio >= 0, "Parameter 'max_missing_ratio' must be larger or equal to 0"

        if isinstance(k, numbers.Number):
            assert k > 0, "Parameter 'k' must have a value of 1 or larger"
            assert k < len(data), "Parameter 'k' must be smaller of equal to the number of instances"

        # Set class attributes
        self.data = data
        self.k = k
        self.max_missing_ratio = max_missing_ratio
        self.is_fit = True

        return self

    def transform(self):
        """
        Transforming data using the fitted TabularPreprocessingPipeline
        :return: pandas.DataFrame with the fitted data
        """

        # Ensure pipeline instance has been fitted
        assert self.is_fit is True, ".fit() has to be called before transforming any data"

        # Removal of instances with only missing values
        data = self.data.dropna(how="all", axis="rows")

        # Remove features with more than 20 % missing values
        data = self._remove_partially_missing(axis="columns")

        # Fill missing values
        if self.k == "standard":
            k = int(np.round(len(data) / 20, 0)) if np.round(len(data) / 20, 0) > 3 else 3
        imputer = KNNImputer(n_neighbors=k)
        data[:] = imputer.fit_transform(data)

        # Removal of correlated features
        # TODO: insert solution from MUW workstation

        return data

    def fit_transform(self, data, k="standard", max_missing_ratio=0.2):
        """
        Standard preprocessing pipeline returning the preprocessed data.
        Includes removal of instances with only missing values, removal of features with more than 20 % missing values,
        kNN-based imputation for missing values, and removal of correlated features.
        :param data: pandas.DataFrame containing the data to be preprocessed
        :param k: int indicating the k nearest neighbors for kNN-based  imputation
        :param max_missing_ratio: float indicating the maximum ratio of missing to all feature values to keep a feature
        :return: pandas.DataFrame containing the preprocessed data
        """

        # Fit
        self.fit(data=data, k=k, max_missing_ratio=max_missing_ratio)

        # Transform
        data = self.transform()

        return data


def sensitivity(cm):
    """
    Calculate sensitivity (=Recall/True positive rate) for confusion matrix with any number of classes
    :param cm: numpy.ndarray with 2 dimensions indicating the confusion matrix
               CAVE: Assumes confusion matrix orientation as given by sklearn.metrics.confusion_matrix with actual
                     classes on the x axis and predictions on the y axis
               CAVE: For binary classifications, label at index 0 in the confusion matrix is regarded as negative class
                     and label at index 1 as positive class
    :return: Float between 0 and 1 indicating the sensitivity
    """

    # Binary classification
    if cm.shape == (2, 2):
        sns = cm[1][1] / (cm[1][1] + cm[1][0])

    # Multiclass classification
    else:
        classwise_sns = []
        for i in np.arange(len(cm)):
            tp = cm[i][i]
            fn_and_tp = cm[i, :]
            if np.sum(fn_and_tp) == 0:
                classwise_sns.append(0)
            else:
                classwise_sns.append(tp / np.sum(fn_and_tp))
            sns = np.sum(classwise_sns) / len(classwise_sns)

    return sns


def specificity(cm):
    """
    Calculate specificity (=True negative rate) for confusion matrix with any number of classes
    :param cm: numpy.ndarray with 2 dimensions indicating the confusion matrix
               CAVE: Assumes confusion matrix orientation as given by sklearn.metrics.confusion_matrix with actual
                     classes on the x axis and predictions on the y axis
               CAVE: For binary classifications, label at index 0 in the confusion matrix is regarded as negative class
                     and label at index 1 as positive class
    :return: Float between 0 and 1 indicating the sensitivity
    """

    # Binary classification
    if cm.shape == (2, 2):
        spc = cm[0][0] / (cm[0][0] + cm[0][1])

    # Multiclass classification
    else:
        classwise_spc = []
        for i in np.arange(len(cm)):
            t = [cm[j][j] for j in np.arange(len(cm))]
            del t[i]
            tn = np.sum(t)
            f = list(cm[i])
            del f[i]
            fp = np.sum(f)
            if (tn + fp) == 0:
                classwise_spc.append(0)
            else:
                classwise_spc.append(tn / (tn + fp))
        spc = np.sum(classwise_spc) / len(classwise_spc)

    return spc


def main():
    # Set hyperparameters
    num_folds = 100
    label_name = "status"

    # Specify data location
    data_path = "/home/clemens/Classification_blood71/clinical_wlabels.csv"

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

    # Convert non-bleeding labels (2, 3) to non-event (0)
    # df = convert_to_bleeding_prediction(df)

    # Display bar chart with number of samples per class
    sns.countplot(x="status", data=df)
    plt.title("Original class frequencies")
    plt.savefig("Results/original_class_frequencies.png")
    plt.close()

    # Separate data into training and test
    y = df["status"]
    x = df.drop("status", axis="columns")

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
    clfs_accs = []
    clfs_aucs = []
    clfs_sns = []
    clfs_spc = []

    # Initialize result table
    results = pd.DataFrame(index=list(clfs.keys()))

    # Iterate over classifiers
    for clf in clfs:

        # Initialize fold-wise and overall performance containers
        cms = np.zeros((num_classes, num_classes))
        accs = []
        aucs = []

        # Iterate over MCCV
        for fold_index in np.arange(num_folds):

            # Split into training and test data
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.15,
                                                                stratify=y,
                                                                random_state=fold_index)

            # Random undersampling
            rus = RandomUnderSampler(random_state=fold_index, sampling_strategy=0.3)
            x_train, y_train = rus.fit_resample(x_train, y_train)

            # SMOTE
            smote = SMOTE(random_state=fold_index, sampling_strategy=0.8)
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
            auc = roc_auc_score(y_test, y_pred)

            # Append performance to fold-wise and overall containers
            cms += cm
            accs.append(acc)
            aucs.append(auc)

        # Calculate overall performance
        avg_acc = np.sum(accs) / len(accs)
        avg_auc = np.sum(aucs) / len(aucs)

        # Append performance to classifier overall performances
        clfs_accs.append(np.round(avg_acc, 2))
        clfs_aucs.append(np.round(avg_auc, 2))
        clfs_sns.append(np.round(sensitivity(cms), 2))
        clfs_spc.append(np.round(specificity(cms), 2))

        # Display overall performances
        print("== {} ==".format(clf))
        print("Cumulative CM:\n", cms)
        print("Avg ACC:", avg_acc)
        print("Avg AUC:", avg_auc)
        print("Avg SNS:", mu(cms))
        print("Avg SPC:", specificity(cms))
        print()

        # Display confusion matrix
        sns.heatmap(cms, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("{} - Confusion matrix".format(clf))
        plt.savefig("Results/confusion_matrix-{}.png".format(clf))
        plt.close()

    # Append performance to result table
    results["ACC"] = clfs_accs
    results["AUC"] = clfs_aucs
    results["SPC"] = clfs_spc
    results["SNS"] = clfs_sns

    # Save result table
    results.to_csv("Results/performances.csv", sep=";")
    results.plot.bar(rot=45).legend(loc="upper right")
    plt.savefig("Results/performance.png".format(clf))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
