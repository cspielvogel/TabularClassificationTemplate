#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:52 2021

Performance metrics for tabular data classification

Content:
    - Performance estimation using Monte Carlo cross validation with multiple metrics
        - AUC
        - Sensitivity
        - Specificity
        - TODO: Accuracy

@author: cspielvogel
"""

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def sensitivity(y_true, y_pred):
    """
    Calculate sensitivity (=Recall/True positive rate) for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :return: Float between 0 and 1 indicating the sensitivity
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

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


def specificity(y_true, y_pred):
    """
    Calculate specificity (=True negative rate) for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :return: Float between 0 and 1 indicating the sensitivity
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

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


def roc_auc(y_true, y_pred, average="macro"):
    """
    Calculation of area under the receiver operating characteristic curve (ROC AUC)
    Based on Afsan Abdulali Gujarati's solution at
        https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param average: String 'micro', 'macro', 'samples' or 'weighted'; default is 'macro'
                    If None, the scores for each class are returned. Otherwise, this determines the type of averaging
                    performed on the data: Note: multiclass ROC AUC currently only handles the 'macro' and 'weighted'
                    averages.
                    'micro':
                    Calculate metrics globally by considering each element of the label indicator matrix as a label.
                    'macro':
                    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                     into account.
                    'weighted':
                    Calculate metrics for each label, and find their average, weighted by support (the number of true
                    instances for each label).
                    'samples':
                    Calculate metrics for each instance, and find their average.
                    See also https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    :return:
    """
    unique_class = set(y_true)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]

        y_true_new = [0 if x in other_class else 1 for x in y_true]
        y_pred_new = [0 if x in other_class else 1 for x in y_pred]

        roc_auc = roc_auc_score(y_true_new, y_pred_new, average=average)
        roc_auc_dict[per_class] = roc_auc

    avg_auc = np.sum(list(roc_auc_dict.values())) / len(roc_auc_dict.values())

    return avg_auc
