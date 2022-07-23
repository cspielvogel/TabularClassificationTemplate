#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jul 13 12:16 2022

Calibration of predicted classifier probabilities

@author: cspielvogel
"""

import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go


def create_calibration_curve(original_model, calibrated_model, features, labels, calibration_path, clf_name, show=False):
    """
    Visualization of classifier calibration via calibration curve

    :param original_model: Uncalibrated model with .predict_proba() function such as sklearn.base.BaseEstimator
    :param calibrated_model: Calibrated model with .predict_proba() function such as sklearn.base.BaseEstimator
    :param features: numpy.ndarray 2D containing the features for each sample
    :param labels: numpy.ndarray 1D containing the labels associated to the samples in the same order as for features
    :param calibration_path: String indicating the path where calibration results shall be saved
    :param clf_name: String indicating classifier name
    :param show: Bool indicating whether to show the calibration curve during run
    :return: None
    """

    # Get predicted probabilities for
    original_probs = original_model.predict_proba(features)[:, 1]
    fop_orig, mpv_orig = calibration_curve(labels.values, original_probs, n_bins=10, normalize=True)

    calibrated_probs = calibrated_model.predict_proba(features)[:, 1]
    fop_calib, mpv_calib = calibration_curve(labels.values, calibrated_probs, n_bins=10, normalize=True)

    orig = pd.DataFrame()
    orig["Fraction of positives (Positive class: 1)"] = fop_orig
    orig["Mean predicted probability (Positive class: 1)"] = mpv_orig
    orig["Model"] = "Original"

    calib = pd.DataFrame()
    calib["Fraction of positives (Positive class: 1)"] = fop_calib
    calib["Mean predicted probability (Positive class: 1)"] = mpv_calib
    calib["Model"] = "Calibrated"

    # Add line indicating perfect calibration
    perfect = pd.DataFrame()
    perfect["Fraction of positives (Positive class: 1)"] = [0, 1]
    perfect["Mean predicted probability (Positive class: 1)"] = [0, 1]
    perfect["Model"] = "Perfectly calibrated"

    # fig = px.line(pd.concat([orig, calib, perfect]),
    #               x="Mean predicted probability (Positive class: 1)",
    #               y="Fraction of positives (Positive class: 1)",
    #               color="Model",
    #               symbol="Model")

    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines+markers",
                             x=orig["Mean predicted probability (Positive class: 1)"],
                             y=orig["Fraction of positives (Positive class: 1)"],
                             name=orig["Model"].values[0],
                             line=dict(color="grey", width=4),
                             marker=dict(size=8),
                             ))
    fig.add_trace(go.Scatter(mode="lines+markers",
                             x=calib["Mean predicted probability (Positive class: 1)"],
                             y=calib["Fraction of positives (Positive class: 1)"],
                             name=calib["Model"].values[0],
                             line=dict(color="dodgerblue", width=4),
                             marker=dict(size=8),
                             ))
    fig.add_trace(go.Scatter(mode="lines",
                             x=perfect["Mean predicted probability (Positive class: 1)"],
                             y=perfect["Fraction of positives (Positive class: 1)"],
                             name=perfect["Model"].values[0],
                             line=dict(color="lightgrey", width=4, dash="dot")
                             ))

    fig.update_layout(title="Calibration",
                      xaxis_title="Mean predicted probability (Positive class: 1)",
                      yaxis_title="Fraction of positives (Positive class: 1)")

    if show is True:
        fig.show()

    fig.write_html(os.path.join(calibration_path, f"calibration_curve_{clf_name}.html"))


def calibrate(model, features, labels, calibration_path, clf_name, verbose=True):
    """
    Calibrate the probabilities of the given model based on the provided data using an ensemble of classifiers
    calibrated on k-fold cross validation folds.

    :param model: Model with .predict_proba() function such as sklearn.base.BaseEstimator
    :param features: numpy.ndarray 2D containing the features for each sample
    :param labels: numpy.ndarray 1D containing the labels associated to the samples in the same order as for features
    :param calibration_path: String indicating the path where calibration results shall be saved
    :param clf_name: String indicating classifier name
    :param verbose: Bool indicating whether run details shall be displayed in output
    :return: Calibrated model
    """

    # # TODO: Implement EBM support
    # if clf_name == "ebm":
    #     print(f"[Warning] Calibration is currently not supported and therefore not performed for {clf_name}")
    #     return model

    # Set calibration method depending on data set size as suggested (scikit-learn.org/stable/modules/calibration.html)
    if len(labels) < 1000:
        calibration_method = "sigmoid"
    else:
        calibration_method = "isotonic"  # TODO: also apply for imbalanced classifications? https://scikit-learn.org/stable/modules/calibration.html

    if verbose is True:
        print(f"[Calibration] Setting calibration method to {calibration_method}")

    # Perform classifier calibration using ensemble built on k-fold CV folds
    calibrated_model = CalibratedClassifierCV(base_estimator=model,
                                              cv=3,
                                              ensemble=True,
                                              method=calibration_method)

    print(features)

    calibrated_model.fit(features, labels)

    # Check whether performing binary classification
    if len(np.unique(labels)) == 2:
        create_calibration_curve(original_model=model,
                                 calibrated_model=calibrated_model,
                                 features=features,
                                 labels=labels,
                                 calibration_path=calibration_path,
                                 clf_name=clf_name,
                                 show=False)

    return calibrated_model


def main():
    pass


if __name__ == "__main__":
    main()
