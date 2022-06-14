#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 06 13:09 2022

Functionality associated to explainability

Includes:
    - Partial dependence plots (DPD)
    - Permutation feature importance (for train and test data)
    - SHAP value computation and summary plotting

TODO:
    - Additional SHAP plots (e.g. heatmap)

@author: cspielvogel
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
import shap

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os


def plot_partial_dependences(model, x, y, feature_names, clf_name, save_path):
    """
    Create and plot partial dependence plots (PDP)

    :param model: sklearn.base.BaseEstimator or derivative containing a trained model
    :param x: numpy.ndarray containing the feature values
    :param y: numpy.ndarray 1D or a list containing the label values
    :param feature_names: numpy.ndarray 1D or a list containing the feature names as strings
    :param clf_name: str indicating the classifiers name
    :param save_path: str indicating the directory path where the outputs shall be saved
    :return: None
    """

    # Get number of classes
    num_classes = len(np.unique(y))

    # Iterate through features and create PDP for each
    for feature_index, _ in enumerate(np.arange(len(feature_names))):

        # Multi class
        if num_classes > 2:

            # Create PDP for each target class
            for target_index in np.arange(num_classes):
                PartialDependenceDisplay.from_estimator(model.best_estimator_,
                                                        X=x,
                                                        features=[feature_index],
                                                        feature_names=feature_names,
                                                        target=np.unique(y)[target_index])
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(save_path,
                                         f"partial_dependence-{clf_name}_feature-{feature_names[feature_index]}_class-{np.unique(y)[target_index]}.png"),
                            bbox_inches="tight")
                plt.close()

        # Single class
        else:
            PartialDependenceDisplay.from_estimator(model.best_estimator_,
                                                    X=x,
                                                    features=[feature_index],
                                                    feature_names=feature_names)
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(os.path.join(save_path,
                                     f"partial_dependence-{clf_name}_feature-{feature_names[feature_index]}.png"),
                        bbox_inches="tight")
            plt.close()


def plot_shap_features(model, x, feature_names, index_names, clf_name, save_path, verbose=True):
    """
    Compute SHAP values, save to file and create summary plots

    :param model: sklearn.base.BaseEstimator or derivative containing a trained model
    :param x: numpy.ndarray containing the feature values
    :param feature_names: numpy.ndarray 1D or a list containing the feature names as strings
    :param index_names: numpy.ndarray 1D or a list containing the sample names
    :param clf_name: str indicating the classifiers name
    :param save_path: str indicating the directory path where the outputs shall be saved
    :param verbose: bool indicating whether commandline output shall be displayed
    :return: pandas.DataFrame containing the SHAP values
    """

    # SHAP analysis
    if verbose is True:
        print("[XAI] Computing SHAP importances")

    # Ensure plotting summary as bar for multiclass and beeswarm for binary classification
    classes = model.best_estimator_.classes_
    if len(classes) > 2:
        predictor = model.best_estimator_.predict_proba
    else:
        predictor = model.best_estimator_.predict

    # Compute SHAP values
    explainer = shap.KernelExplainer(predictor, x)
    shap_values = explainer.shap_values(x)

    # Save SHAP values to file
    if len(classes) == 2:
        shap_df = pd.DataFrame(shap_values,
                               columns=feature_names,
                               index=index_names)
        shap_df.to_csv(os.path.join(save_path, f"shap-values_{clf_name}.csv"), sep=";")
    else:
        for i, label in enumerate(classes):
            shap_df = pd.DataFrame(shap_values[i],
                                   columns=feature_names,
                                   index=index_names)
            shap_df.to_csv(os.path.join(save_path, f"Label-{label}_shap-values.csv"), sep=";")

    shap.summary_plot(shap_values=shap_values,
                      features=x,
                      feature_names=feature_names,
                      class_names=model.best_estimator_.classes_,
                      show=False)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_path, f"shap_summary-{clf_name}.png"),
                bbox_inches="tight")
    plt.close()

    return shap_df


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
    :param str save_path: Indicate path to save plots and result files to; If None, no files are saved
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
        plt.close()

    if display_plots is True:
        # Display plot
        plt.show()

    # Save importances to file
    if save_path is True:
        importance_df.to_csv(save_path, sep=";")

    return importance_df


def main():
    pass


if __name__ == "__main__":
    main()
