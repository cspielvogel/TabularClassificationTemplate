#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:46 2021

Exploratory data analysis

Includes:
    - Pandas profiling
    - UMAP

# TODO: add docstrings
# TODO: add tSNE and PCA

@author: cspielvogel
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px

from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from preprocessing import TabularIntraFoldPreprocessor


def run_pandas_profiling(data, save_path, verbose=True):
    if verbose is True:
        print("[EDA] Starting Pandas Profiling")

    profile = ProfileReport(data, title="Pandas Profiling Report", minimal=True)
    profile.to_file(os.path.join(save_path, "exploratory_data_analysis.html"))


def run_umap(features, labels, label_column, save_path):

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


def run_tsne(features, labels, label_column, save_path):
    pass


def run_pca(features, labels, label_column, save_path):
    pass


def run_eda(features, labels, label_column, save_path, analyses_to_run=("pandas_profiling", "umap", "tsne", "pca"),
            verbose=True):
    if "pandas_profiling" in analyses_to_run:
        data = features.copy()
        data[label_column] = labels.copy()

        run_pandas_profiling(data=data,
                             save_path=save_path,
                             verbose=verbose)

    if "umap" in analyses_to_run or "tsne" in analyses_to_run or "pca" in analyses_to_run:

        # Standardize and impute missing values
        features_standardized = TabularIntraFoldPreprocessor().fit_transform(features)

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
