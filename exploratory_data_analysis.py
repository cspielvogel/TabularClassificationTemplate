#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:46 2021

Exploratory data analysis

Includes:
    - Pandas profiling
    - UMAP
    - tSNE
    - PCA

# TODO: add dimensionality reduction algorithm parameters
# TODO: add option to pass dictionary with parameters to dim reduction algorithms

@author: cspielvogel
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px

from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

from preprocessing import TabularIntraFoldPreprocessor


def run_pandas_profiling(data, save_path, verbose=True):
    """
    Perform exploratory data analysis using pandas profiling and save corresponding report as HTML file

    :param data: pandas.DataFrame containing the data for EDA
    :param save_path: str indicating the directory where the EDA report shall be saved
    :param verbose: bool indicating whether commandline output shall be shown
    :return: None
    """
    if verbose is True:
        print("[EDA] Starting Pandas Profiling")

    profile = ProfileReport(data, title="Pandas Profiling Report", minimal=True)
    profile.to_file(os.path.join(save_path, "exploratory_data_analysis.html"))


def run_umap(features, labels, label_column, save_path):
    """
    Perform uniform manifold approximation and projection for dimensionality reduction to two dimensions and plot

    :param features: numpy.ndarray containing the feature values
    :param labels: numpy.ndarray 1D or a list containing the label values
    :param label_column: str indicating the name of the label column (used for plotting)
    :param save_path: str indicating the directory path where the result plot will be saved
    :return: numpy.ndarray containing the embedded features
    """

    # Reduce to 2 dimensions using UMAP
    reducer = UMAP()    # TODO: Set parameters
    embeddeding = reducer.fit_transform(features)

    data_reduced = pd.DataFrame()
    data_reduced["UMAP1"] = embeddeding[:, 0]
    data_reduced["UMAP2"] = embeddeding[:, 1]

    # Append label column to data table for visualization
    data_reduced[label_column] = labels.astype(str).values

    # 2D scatter plot
    scatter1 = px.scatter(data_reduced,
                          x="UMAP1",
                          y="UMAP2",
                          color=label_column,
                          labels={label_column: "Annotation"},
                          color_discrete_sequence=px.colors.qualitative.Dark24,  # plotly.com/python/discrete-color/
                          width=900,
                          height=800,
                          marginal_x="histogram",
                          marginal_y="histogram",
                          template="plotly_white")  # plotly.com/python/templates/ e.g. simple_white, plotly_dark
    scatter1.update_traces(marker=dict(size=12,
                                       opacity=0.4,
                                       line=dict(width=1,
                                                 color="black")),
                           selector=dict(mode="markers"))

    # Save plot to HTML file
    scatter1.write_html(os.path.join(save_path, "umap.html"))

    return data_reduced


def run_tsne(features, labels, label_column, save_path):
    """
    Perform t-stochastic neighbor embedding for dimensionality reduction to two dimensions and plot

    :param features: numpy.ndarray containing the feature values
    :param labels: numpy.ndarray 1D or a list containing the label values
    :param label_column: str indicating the name of the label column (used for plotting)
    :param save_path: str indicating the directory path where the result plot will be saved
    :return: numpy.ndarray containing the embedded features
    """

    # Reduce to 2 dimensions using UMAP
    reducer = TSNE()    # TODO: Set parameters
    embeddeding = reducer.fit_transform(features)

    data_reduced = pd.DataFrame()
    data_reduced["tSNE1"] = embeddeding[:, 0]
    data_reduced["tSNE2"] = embeddeding[:, 1]

    # Append label column to data table for visualization
    data_reduced[label_column] = labels.astype(str).values

    # 2D scatter plot
    scatter1 = px.scatter(data_reduced,
                          x="tSNE1",
                          y="tSNE2",
                          color=label_column,
                          labels={label_column: "Annotation"},
                          color_discrete_sequence=px.colors.qualitative.Dark24,  # plotly.com/python/discrete-color/
                          width=900,
                          height=800,
                          marginal_x="histogram",
                          marginal_y="histogram",
                          template="plotly_white")  # plotly.com/python/templates/ e.g. simple_white, plotly_dark
    scatter1.update_traces(marker=dict(size=12,
                                       opacity=0.4,
                                       line=dict(width=1,
                                                 color="black")),
                           selector=dict(mode="markers"))

    # Save plot to HTML file
    scatter1.write_html(os.path.join(save_path, "tsne.html"))

    return data_reduced


def run_pca(features, labels, label_column, save_path):
    """
    Perform principal component analysis for dimensionality reduction to two dimensions and plot

    :param features: numpy.ndarray containing the feature values
    :param labels: numpy.ndarray 1D or a list containing the label values
    :param label_column: str indicating the name of the label column (used for plotting)
    :param save_path: str indicating the directory path where the result plot will be saved
    :return: numpy.ndarray containing the embedded features
    """

    # Reduce to 2 dimensions using UMAP
    reducer = PCA()    # TODO: Set parameters
    embeddeding = reducer.fit_transform(features)

    data_reduced = pd.DataFrame()
    data_reduced["PCA1"] = embeddeding[:, 0]
    data_reduced["PCA2"] = embeddeding[:, 1]

    # Append label column to data table for visualization
    data_reduced[label_column] = labels.astype(str).values

    # 2D scatter plot
    scatter1 = px.scatter(data_reduced,
                          x="PCA1",
                          y="PCA2",
                          color=label_column,
                          labels={label_column: "Annotation"},
                          color_discrete_sequence=px.colors.qualitative.Dark24,  # plotly.com/python/discrete-color/
                          width=900,
                          height=800,
                          marginal_x="histogram",
                          marginal_y="histogram",
                          template="plotly_white")  # plotly.com/python/templates/ e.g. simple_white, plotly_dark
    scatter1.update_traces(marker=dict(size=12,
                                       opacity=0.4,
                                       line=dict(width=1,
                                                 color="black")),
                           selector=dict(mode="markers"))

    # Save plot to HTML file
    scatter1.write_html(os.path.join(save_path, "pca.html"))

    return data_reduced


def run_eda(features, labels, label_column, save_path, analyses_to_run=("pandas_profiling", "umap", "tsne", "pca"),
            verbose=True):
    """
    Perform exploratory data analysis

    :param features: numpy.ndarray containing the feature values
    :param labels: numpy.ndarray 1D or a list containing the label values
    :param label_column: str indicating the name of the label column (used for plotting)
    :param save_path: str indicating the directory path where the result plot will be saved
    :param analyses_to_run: tuple containing strings indicating the techniques to be executed
    :param verbose: bool indicating whether to display commandline output
    :return: None
    """

    if "pandas_profiling" in analyses_to_run:
        data = features.copy()
        data[label_column] = labels.copy()

        run_pandas_profiling(data=data,
                             save_path=save_path,
                             verbose=verbose)

    if "umap" in analyses_to_run or "tsne" in analyses_to_run or "pca" in analyses_to_run:

        # Standardize and impute missing values
        features_standardized = TabularIntraFoldPreprocessor(imputation_method="knn",
                                                             k="automated",
                                                             normalization="standardize").fit_transform(features)

    if "umap" in analyses_to_run:
        run_umap(features=features_standardized, labels=labels, label_column=label_column, save_path=save_path)
    if "tsne" in analyses_to_run:
        run_tsne(features=features_standardized, labels=labels, label_column=label_column, save_path=save_path)
    if "pca" in analyses_to_run:
        run_pca(features=features_standardized, labels=labels, label_column=label_column, save_path=save_path)


def main():
    pass


if __name__ == "__main__":
    main()
