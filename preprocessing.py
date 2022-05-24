#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:46 2021

Preprocessing functionalities for tabular data classification

Content:
    - Preprocessing pipeline (to be applied before train/val/test splitting)
        - Removing all-NA instances
        - Remove constant features
        - Removing features with too many missing values (default > 20% NaNs)
        - TODO: removal of correlated features (foldwise)
        - TODO: remove samples that have all NA except for label (dropna with thres? for percent missing?)
        - TODO: remove samples that have no label
        - TODO: automatically detect categorical features and convert those to one-hot while normalizing numeric features

    - Fold-wise preprocessing pipeline
        - Normalization (standardization per default)
        - Filling missing values using kNN imputation
        - TODO:Resampling e.g. SMOTE
        - Normalize only numeric features and one-hot encode categorical features

@author: cspielvogel
"""

import warnings
import numbers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from pandas.core.common import SettingWithCopyWarning

# Ignore SettingWithCopyWarning resulting from creating pandas.DataFrames from numpy.ndarrays
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class TabularPreprocessor:
    """
    Standardized preprocessing pipeline to be used before any training / validation / test splitting procedure is
    applied
    """

    def __init__(self, max_missing_ratio=0.2):
        """
        Constructor
        :param max_missing_ratio: float indicating the maximum ratio of missing to all feature values to keep a feature
        """

        # Ensure input parameter validity
        assert isinstance(max_missing_ratio, numbers.Number), "Parameter 'max_missing_ratio' must be float"
        assert max_missing_ratio <= 1, "Parameter 'max_missing_ratio' must be less or equal 1"
        assert max_missing_ratio >= 0, "Parameter 'max_missing_ratio' must be larger or equal to 0"

        # Set attributes
        self.data = None
        self.label_name = None
        self.max_missing_ratio = max_missing_ratio
        self.is_fit = False

    def _remove_partially_missing(self, axis="columns"):
        """
        Removal of features or instances with missing features above the given ratio
        :param axis: string (must be one of "rows" or "columns) indicating whether to remove rows or columns
        :return: pandas.DataFrame without rows or columns with missing values above the max_missing_ratio
        """

        # Ensure valid parameters
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

    def fit(self, data, label_column):
        """
        Fitting the standard preprocessing pipeline before transformation.
        Includes removal of instances with only missing values, removal of features with more than 20 % missing values,
        kNN-based imputation for missing values, and removal of correlated features.
        :param data: pandas.DataFrame containing the data to be preprocessed
        :param label_column: str indicating the column name containing the label
        :return: TabularPreprocessor fitted with the given parameters
        """

        # Ensure valid parameters
        assert isinstance(data, pd.DataFrame), "Parameter 'data' must be an instance of pandas.DataFrame"

        # Assign data to attribute
        self.data = data

        # Assign label to attribute
        self.label_name = label_column

        # Set flag indicating conducted fitting
        self.is_fit = True

        return self

    def transform(self):
        """
        Transforming data using the fitted TabularPreprocessor
        :return: pandas.DataFrame with the fitted data
        """

        # Ensure pipeline instance has been fitted
        assert self.is_fit is True, ".fit() has to be called before transforming any data"

        # Only keep first instance if multiple instances have the same key
        num_instances_before = len(self.data)
        self.data = self.data[~self.data.index.duplicated(keep="first")]
        num_instances_diff = num_instances_before - len(self.data)
        if num_instances_diff > 0:
            print(f"[Warning] {num_instances_diff} instance(s) removed due to duplicate keys"
                  f"- only keeping first occurrence!")

        # Remove instances with missing label
        num_label_nans = self.data[self.label_name].isnull().sum()
        self.data.dropna(subset=[self.label_name], inplace=True)

        if num_label_nans > 0:
            print(f"[Warning] {num_label_nans} sample(s) removed due to missing label!")

        # Removal of instances with only missing values
        self.data = self.data.dropna(how="all", axis="rows")

        # Remove features with more than given percentage of missing values (self.max_missing_ratio)
        self.data = self._remove_partially_missing(axis="columns")

        # Remove features with constant value over all instances while ignoring NaNs
        for column in self.data.columns:
            if self.data[column].dtype == "int64":
                if np.all(self.data[column][~np.isnan(self.data[column])].values ==
                          self.data[column][~np.isnan(self.data[column])].values[0]):
                    self.data = self.data.drop(column, axis="columns")
            else:
                if np.all(np.array([i for i in self.data[column] if not i in ['nan', np.nan]]) ==
                          self.data[column].values[0]):
                    self.data = self.data.drop(column, axis="columns")

        return self.data

    def fit_transform(self, data, label_name):
        """
        Standard preprocessing pipeline returning the preprocessed data.
        Includes removal of instances with only missing values, removal of features with more than 20 % missing values,
        kNN-based imputation for missing values, and removal of correlated features.
        :param data: pandas.DataFrame containing the data to be preprocessed
        :param label_name: str indicating column name containing label
        :return: pandas.DataFrame containing the preprocessed data
        """

        # Fit
        self.fit(data=data, label_column=label_name)

        # Transform
        data = self.transform()

        return data


class TabularIntraFoldPreprocessor:
    """
    Preprocessing pipeline to be conducted for each split / fold in the validation procedure individually to avoid any
    data leakage

    Performs
    - k-nearest neighbor-based feature imputation
    - Standardization
    - TODO: SMOTE
    """

    def __init__(self, k="automated", normalization="standardize"):
        """
        Constructor
        :param k: int indicating the k nearest neighbors for kNN-based imputation
        :param normalization: string indicating the typ of normalization, must be one of "standardize", "minmax"
        :return: None
        """

        # Ensure input parameter validity
        assert isinstance(k, numbers.Number) or k == "automated", "Parameter 'k' must either be numeric or 'automated'"
        assert normalization in ["standardize", "minmax"], \
            "Parameter 'normalization' must be one of ('standardize', 'minmax')"

        # Set attributes
        self.k = k  # Number of nearest neighbors for kNN imputation
        self.normalization = normalization  # Type of normalization to carry out
        self.is_fit = False

        # Initialize attributes set during processing
        self.data = None    # Data used for fitting
        self.scaler = None
        self.imputer = None

    def fit(self, data):
        """
        Fitting the standard preprocessing pipeline before transformation. Includes standardization and kNN-based
        imputation of missing feature values
        :param data: pandas.DataFrame containing the per-fold training data used for fitting
        :return: TabularPreprocessor fitted with the given parameters
        """

        # Ensure valid parameters
        assert isinstance(data, pd.DataFrame), "Parameter 'data' must be an instance of pandas.DataFrame"

        # Ensure that the number of nearest neighbors used for imputation is above zero if not a string
        if isinstance(self.k, numbers.Number):
            assert self.k > 0, "Parameter 'k' must have a value of 1 or larger"
            assert self.k < len(self.data), "Parameter 'k' must be smaller of equal to the number of instances"

        # Set data to attribute
        self.data = data

        # Normalize features
        if self.normalization == "standardize":
            scaler = StandardScaler()
        elif self.normalization == "minmax":
            scaler = MinMaxScaler()
        self.scaler = scaler.fit(self.data)

        # Fill missing values
        if self.k == "automated":    # Use rounded down number of samples divided by 20 but at least 3 as k
            k = int(np.round(len(self.data) / 20, 0)) if np.round(len(self.data) / 20, 0) > 3 else 3
        imputer = KNNImputer(n_neighbors=k)
        self.imputer = imputer.fit(self.data)

        # Set flag to indicate fitting was conducted
        self.is_fit = True

        return self

    def transform(self, data=None):
        """
        Transforming data using the fitted TabularIntraFoldPreprocessor
        :param data: pandas.DataFrame containing data to transform using the fitted models, use data from fitting
        procedure if None
        :return: pandas.DataFrame containing the transformed data table
        """

        # Ensure pipeline instance has been fitted
        assert self.is_fit is True, ".fit() has to be called before transforming any data"

        # If no data given, use data from fitting procedure
        if data is None:
            data = self.data

        # Ensure data is of type pandas.DataFrame
        assert isinstance(data, pd.DataFrame), "Parameter 'data' must be an instance of pandas.DataFrame"

        # Apply feature scaler
        data[:] = self.scaler.transform(data)

        # Apply kNN-imputation
        data[:] = self.imputer.transform(data)

        return data

    def fit_transform(self, data=None):
        """
        Standard preprocessing pipeline returning the preprocessed data for individual folds.
        Includes removal of instances with only missing values, removal of features with more than 20 % missing values,
        kNN-based imputation for missing values, and removal of correlated features.
        :param data: pandas.DataFrame containing data to transform using the fitted models, use data from fitting
        procedure if None
        :return: pandas.DataFrame containing the preprocessed data
        """

        # Fit pipeline components
        self.fit(data=data)

        # Transform data using the fitted pipeline
        data = self.transform(data)

        return data
