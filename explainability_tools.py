#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 06 13:09 2022

Functionality associated to explainability
Includes:
    - Permutation feature importance (for train and test data)
TODO:
    - Partial dependence plots

@author: cspielvogel
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os


def plot_importances(importances_mean, importances_std, feature_names, plot_title, order_alphanumeric=True,
                     include_top=0, display_plots=True, save_path=None):
    """
    Create barplot of feature importances

    :param numpy.array importances_mean: Indicating mean importances for each feature over all folds in a model
    :param numpy.array importances_std: Indicating mean standard deviation for each feature over all folds in a model
    :param numpy.array feature_names: Indicting name of feature variables
    :param str plot_title: Indicating title as header over plot
    :param bool order_alphanumeric: Indicating whether to sort features alphanumerically in barplot; Keeps order if False
    :param int include_top: Number of features to include in the plot ordered by highest importance; All if 0
    :param bool display_plots: If True, show plots on creation time
    :param str save_path: Indicate path to save plots to; If None, plots are not saved
    :return: pandas.DataFrame with formatted mean and standard deviation for feature importances
    """

    # Format importances to dataframe
    importance_df = pd.DataFrame()
    importance_df["Mean"] = importances_mean
    importance_df["SD"] = importances_std
    importance_df["Feature"] = feature_names

    if include_top > 0:
        # Remove features not included in best features by mean importance
        importance_df.sort_values(by="Mean", ascending=False, inplace=True)
        importance_df = importance_df.head(include_top)

    if order_alphanumeric is True:
        # Alphanumeric sorting
        importance_df.sort_values(by="Feature", inplace=True)

    # Color best features with darker colors
    pal = sns.color_palette("Blues_d", len(importance_df))
    rank = importance_df.Mean.argsort().argsort()

    # Adjust size of horizontal caps of error bars depending on number of features
    adjusted_capsize = 45 / len(importance_df)

    # Plotting barplot with errorbars and axis labels
    ax = sns.barplot(data=importance_df,
                     x="Feature",
                     y="Mean",
                     palette=np.array(pal)[rank])
    ax.errorbar(data=importance_df,
                x="Feature",
                y="Mean",
                yerr="SD",
                ls="",
                lw=3,
                color="black",
                capsize=adjusted_capsize)
    ax.set_title(f"{plot_title}")
    ax.set_ylabel("Importance")
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Avoid x labels being cut off
    plt.gcf().subplots_adjust(bottom=0.50)
    plt.subplots_adjust(bottom=0.50)

    if save_path is not None:
        # Save plot
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
        plt.savefig(save_path)

    if display_plots is True:
        # Display plot
        plt.show()

    return importance_df


def main():
    """
    DEMO
    """

    from sklearn.datasets import load_breast_cancer

    # Set hyperparameters
    num_folds = 10
    plot_title = "10-fold MCCV mean feature importance"
    data = load_breast_cancer()
    # data = load_iris()
    feature_names = data.feature_names

    # Initialize fold-wise feature importance containers for mean and standard deviation with zero values
    raw_importances_foldwise_mean_train = np.zeros((len(feature_names),))
    raw_importances_foldwise_std_train = np.zeros((len(feature_names),))
    raw_importances_foldwise_mean_val = np.zeros((len(feature_names),))
    raw_importances_foldwise_std_val = np.zeros((len(feature_names),))

    # Perform Monate Carlo cross-validation
    for i in range(num_folds):
        x_train, x_val, y_train, y_val = train_test_split(
            data.data, data.target, random_state=i)

        model = RandomForestClassifier(random_state=i).fit(x_train, y_train)

        # Compute and display fold-wise performance
        if len(np.unique(y_train)) > 2:
            y_pred = model.predict_proba(x_val)
        else:
            y_pred = model.predict(x_val)

        model_performance = roc_auc_score(y_val, y_pred, multi_class="ovr")
        print(f"Fold {i+1} AUC: {model_performance:.2}")

        # Compute permutation feature importance scores on training and validation data
        raw_importances_train = permutation_importance(model, x_val, y_val,
                                                       n_repeats=30,
                                                       scoring="roc_auc_ovr",
                                                       n_jobs=10,   # TODO: adjust and automate
                                                       random_state=i)
        raw_importances_val = permutation_importance(model, x_train, y_train,
                                                     n_repeats=30,
                                                     scoring="roc_auc_ovr",
                                                     n_jobs=10,   # TODO: adjust and automate
                                                     random_state=i)

        # Add importance scores to overall container
        raw_importances_foldwise_mean_train += raw_importances_train.importances_mean
        raw_importances_foldwise_std_train += raw_importances_train.importances_std
        raw_importances_foldwise_mean_val += raw_importances_val.importances_mean
        raw_importances_foldwise_std_val += raw_importances_val.importances_std

    # Get mean of feature importance scores and standard deviation over all folds
    overall_mean_importances_train = raw_importances_foldwise_mean_train / num_folds
    overall_std_importances_train = raw_importances_foldwise_std_train / num_folds
    overall_mean_importances_val = raw_importances_foldwise_mean_val / num_folds
    overall_std_importances_val = raw_importances_foldwise_std_val / num_folds

    # Plot feature importances as determined using training and validation data
    plot_importances(importances_mean=overall_mean_importances_train,
                     importances_std=overall_std_importances_train,
                     feature_names=feature_names,
                     plot_title=plot_title + " - Training data",
                     order_alphanumeric=True,
                     include_top=0,
                     display_plots=True,
                     save_path="tmp/")

    plot_importances(importances_mean=overall_mean_importances_val,
                     importances_std=overall_std_importances_val,
                     feature_names=feature_names,
                     plot_title=plot_title + " - Validation data",
                     order_alphanumeric=True,
                     include_top=0,
                     display_plots=True,
                     save_path="tmp/")


if __name__ == "__main__":
    main()
