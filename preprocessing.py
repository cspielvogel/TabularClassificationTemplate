#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:46 2021

Preprocessing functionalities for tabular data classification

Content:
    - Preprocessing pipeline
        - Removing all-NA instances
        - Removing features with too many missing values (>0.2)
        - Filling missing values using kNN imputation
        - TODO: removal of correlated features (foldwise)
        - TODO:Resampling e.g. SMOTE

@author: cspielvogel
"""

import numbers

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer


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
