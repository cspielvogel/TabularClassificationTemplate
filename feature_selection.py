#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 22 10:40 2021

Functionalities for feature selection
Included methods:
    - ANOVA
    - chi squared
    
TODO
    - mRMR feature selection

@author: clemens
"""

import warnings
from math import log2
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def univariate_feature_selection(x_train, y_train, x_test, score_func=f_classif, num_features="log2n"):
    """
    Univariate feature selection using analysis of variance inference test

    :param x_train: numpy.ndarray with 2 dimensions, containing the training feature values
    :param y_train: numpy.ndarray with 1 diimension, containing the training labels
    :param score_func: Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single
        array with scores. See also
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    :param x_test: numpy.ndarray with 2 dimensions, indicating the test feature values
    :param num_features: either int or string, first indicating the number of features to select, second can be either
        "log2n", "sqrtn" or "all" to specify the strategy of deriving the number of selected from total features
    :return: tuple of numpy.ndarray 1D, numpy.ndarray 2D, numpy.ndarray 2D first contains the indices of the selected
        features, second contains the subset of selected training feature values and third the subset of selected test
        features
    """
    
    # Display interpretability warning
    warnings.warn("Filter-based selection requires previous removal of redundant features to ensure interpretability. If you have not done so, perform a removal of redundant features or perform mRMR feature selection.")

    # Specify number of features to select
    if num_features == "log2n":
        num_features = int(np.round(log2(x_train.shape[1])))
    elif num_features == "sqrtn":
        num_features = int(np.round(sqrt(x_train.shape[1])))

    # Initialize feature selector
    selector = SelectKBest(score_func=score_func, k=num_features)

    # Fit and transform training data
    x_train_selected = selector.fit_transform(x_train, y_train)

    # Transform test features
    x_test_selected = selector.transform(x_test)

    # Get indices of selected features
    index_selected = np.array(sorted(selector.scores_.argsort()[-num_features:]))

    return index_selected, x_train_selected, x_test_selected


if __name__ == "__main__":
    main()
