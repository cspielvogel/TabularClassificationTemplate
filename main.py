#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:56 2021

Template for binary classifications of tabular data including preprocessing
# TODO: Implement OO version of main function contents
# TODO: Add one way partial dependence plots and maybe two way for top features
# TODO: Add analysis flow diagram to README
# TODO: Add commandline options /w argparse
# TODO: Test SHAP for multiclass prediction
# TODO: Silence pandas profiling and SHAP commandline output
# TODO: Add pandas profiling before any preprocessing
# TODO: ensure that variable names used in savefig() don't contain special characters which can't be used in file name
# TODO: Add surrogate models
# TODO: Update content below

Content:
    - Feature selection
    - Training and evaluation of multiple classification algorithms
        - k-nearest neighbors
        - Decision tree
        - Random forest
        - Neural network
    - Visualization of performance evaluation
        - Barplot with performances for the classification models

Input data format specifications:
    - As of now, a file path has to be supplied to the main function as string value for the variable "data_path";
    - The file shall be a CSV file separated by a semicolon (;)
    - The file shall have a header, containing the attribute names and the label name
    - The file shall have an index column containing a unique identifier for each instance

@author: cspielvogel
"""

import os
import pickle
import json

import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
from interpret.glassbox import ExplainableBoostingClassifier

from exploratory_data_analysis import run_eda
import metrics
from preprocessing import TabularPreprocessor, TabularIntraFoldPreprocessor
from feature_selection import univariate_feature_selection, mrmr_feature_selection
from explainability_tools import plot_importances


def main():
    # Set hyperparameters
    num_folds = 2
    # label_name = "DPD_final"
    # label_name = "1"
    label_name = "OS_histo36"
    features_to_remove = []
    verbose = True
    classifiers_to_run = ["ebm", "dt", "knn", "rf", "nn"]
    # classifiers_to_run = ["ebm"]
    # output_path = r"C:\Users\cspielvogel\PycharmProjects\TabularClassificationTemplate"
    output_path = r"C:\Users\cspielvogel\PycharmProjects\HNSCC"
    eda_result_path = os.path.join(output_path, r"Results/EDA/")
    explainability_result_path = os.path.join(output_path, r"Results/XAI/")
    model_result_path = os.path.join(output_path, r"Results/Models/")
    performance_result_path = os.path.join(output_path, r"Results/Performance/")
    intermediate_data_path = os.path.join(output_path, r"Results/Intermediate_Data")

    # Specify data location
    # data_path = "/home/cspielvogel/ImageAssociationAnalysisKeggPathwayGroups/Data/Dedicaid/fdb_multiomics_w_labels_bonferroni_significant_publication_OS36.csv"
    # data_path = "/home/cspielvogel/DataStorage/Bone_scintigraphy/Data/umap_feats_pg.csv"
    # data_path = r"C:\Users\cspielvogel\PycharmProjects\TabularClassificationTemplate\Data\test_data.csv"
    data_path = r"C:\Users\cspielvogel\Downloads\fdb_multiomics_w_labels_all.csv"

    # Create save directories if they do not exist yet
    for path in [eda_result_path, explainability_result_path, model_result_path, performance_result_path,
                 intermediate_data_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Load data to table
    df = pd.read_csv(data_path, sep=";", index_col=0)

    # Remove features to be removed as specified
    for col in features_to_remove:
        df = df.drop(col, axis="columns")

    # Perform standardized preprocessing
    preprocessor = TabularPreprocessor()
    df = preprocessor.fit_transform(df, label_name=label_name)

    # Separate data into training and test
    y = df[label_name]
    # y = (df[label_name] < 2) * 1    # TODO: remove; only for PG classification
    x = df.drop(label_name, axis="columns")
    feature_names = x.columns

    # Perform EDA and save results
    run_eda(features=x,
            labels=y,
            label_column=label_name,
            save_path=eda_result_path,
            verbose=verbose)

    # Setup classifiers
    ebm = ExplainableBoostingClassifier()
    ebm_param_grid = {}     # TODO: Add parameter grid (https://interpret.ml/docs/ebm.html)
    ebm_param_grid = {}

    knn = KNeighborsClassifier()
    knn_param_grid = {"weights": ["distance"],
                      "n_neighbors": [int(val) for val in np.round(np.sqrt(x.shape[1])) + np.arange(5) + 1] +
                                     [int(val) for val in np.round(np.sqrt(x.shape[1])) - np.arange(5) if val >= 1],
                      "p": np.arange(1, 5)}
    knn_param_grid = {}

    dt = DecisionTreeClassifier()
    dt_param_grid = {"splitter": ["best", "random"],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    dt_param_grid = {}  # TODO: reactivate parameter grids

    rf = RandomForestClassifier()
    rf_param_grid = {"criterion": ["entropy"],
                     "n_estimators": [500],
                     "max_depth": np.arange(1, 20),
                     "min_samples_split": [2, 4, 6],
                     "min_samples_leaf": [1, 3, 5, 6],
                     "max_features": ["auto", "sqrt", "log2"]}
    rf_param_grid = {}

    nn = MLPClassifier()
    nn_param_grid = {"hidden_layer_sizes": [(32, 64, 32)],
                     "early_stopping": [True],
                     "n_iter_no_change": [20],
                     "max_iter": [1000],
                     "activation": ["relu", "tanh", "logistic"],
                     "solver": ["adam"],
                     "learning_rate_init": [0.01, 0.001, 0.0001]}
    nn_param_grid = {}

    clfs = {"ebm":
                {"classifier": ebm,
                "parameters": ebm_param_grid},
            "knn":
                {"classifier": knn,
                 "parameters": knn_param_grid},
            "dt":
                {"classifier": dt,
                 "parameters": dt_param_grid},
            "rf":
                {"classifier": rf,
                 "parameters": rf_param_grid},
            "nn":
                {"classifier": nn,
                 "parameters": nn_param_grid}}

    clfs_performance = {"acc": [], "sns": [], "spc": [], "auc": []}

    # Get number of classes
    num_classes = len(np.unique(y))

    # Initialize result table
    results = pd.DataFrame(index=classifiers_to_run)

    # Iterate over classifiers
    for clf in clfs:
        if clf not in classifiers_to_run:
            continue

        if verbose is True:
            print(f"[Model training] Starting training for {clf} classifier")

        # Initialize cumulated confusion matrix and fold-wise performance containers
        cms = np.zeros((num_classes, num_classes))
        performance_foldwise = {"acc": [], "sns": [], "spc": [], "auc": []}

        # Iterate over MCCV
        for fold_index in np.arange(num_folds):

            # Split into training and test data
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.15,
                                                                stratify=y,
                                                                shuffle=True,
                                                                random_state=fold_index)

            # Perform standardization and feature imputation
            intra_fold_preprocessor = TabularIntraFoldPreprocessor(k="automated", normalization="standardize")
            intra_fold_preprocessor = intra_fold_preprocessor.fit(x_train)
            x_train = intra_fold_preprocessor.transform(x_train)
            x_test = intra_fold_preprocessor.transform(x_test)

            # Perform feature selection
            selected_indices, x_train, x_test = mrmr_feature_selection(x_train.values,
                                                                       y_train.values,
                                                                       x_test.values,
                                                                       # score_func=f_classif,
                                                                       num_features="log2n")
            feature_names_selected = feature_names[selected_indices]

            # Random undersampling    # TODO: evaluate for performance improvement and reactivate
            # rus = RandomUnderSampler(random_state=fold_index, sampling_strategy=0.4)
            # x_train, y_train = rus.fit_resample(x_train, y_train)

            # SMOTE
            if num_classes == 2:
                smote = SMOTE(random_state=fold_index, sampling_strategy=1)
                x_train, y_train = smote.fit_resample(x_train, y_train)
            else:
                pass    # TODO: enable multi class smote (dict)

            # Setup model
            model = clfs[clf]["classifier"]
            model.random_state = fold_index

            # Hyperparameter tuning and keep model trained with the best set of hyperparameters
            optimized_model = RandomizedSearchCV(model,
                                                 param_distributions=clfs[clf]["parameters"],
                                                 cv=5,
                                                 random_state=fold_index)
            optimized_model.fit(x_train, y_train)

            # Compute and display fold-wise performance
            y_pred = optimized_model.best_estimator_.predict(x_test)

            # Compute permutation feature importance scores on training and validation data
            raw_importances_train = permutation_importance(optimized_model.best_estimator_, x_test, y_test,
                                                           n_repeats=30,
                                                           scoring="roc_auc_ovr",
                                                           n_jobs=10,  # TODO: adjust and automate
                                                           random_state=fold_index)
            raw_importances_val = permutation_importance(optimized_model.best_estimator_, x_train, y_train,
                                                         n_repeats=30,
                                                         scoring="roc_auc_ovr",
                                                         n_jobs=10,  # TODO: adjust and automate
                                                         random_state=fold_index)

            # Initialize fold-wise feature importance containers for mean and standard deviation with zero values
            if "raw_importances_foldwise_mean_train" not in locals():
                raw_importances_foldwise_mean_train = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_std_train = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_mean_val = np.zeros((len(feature_names_selected),))
                raw_importances_foldwise_std_val = np.zeros((len(feature_names_selected),))

            # Add importance scores to overall container
            raw_importances_foldwise_mean_train += raw_importances_train.importances_mean
            raw_importances_foldwise_std_train += raw_importances_train.importances_std
            raw_importances_foldwise_mean_val += raw_importances_val.importances_mean
            raw_importances_foldwise_std_val += raw_importances_val.importances_std

            # Compute performance
            cm = confusion_matrix(y_test, y_pred)
            acc = metrics.accuracy(y_test, y_pred)
            sns = metrics.sensitivity(y_test, y_pred)
            spc = metrics.specificity(y_test, y_pred)
            auc = metrics.roc_auc(y_test, y_pred)

            # Append performance to fold-wise and overall containers
            cms += cm
            performance_foldwise["acc"].append(acc)
            performance_foldwise["sns"].append(sns)
            performance_foldwise["spc"].append(spc)
            performance_foldwise["auc"].append(auc)

        # Setup final model
        seed = 0
        model = clfs[clf]["classifier"]
        model.random_state = seed

        # Preprocess data for creation of final model
        intra_fold_preprocessor = TabularIntraFoldPreprocessor(k="automated", normalization="standardize")
        x_preprocessed = intra_fold_preprocessor.fit_transform(x)

        # Save preprocessed data
        x_preprocessed_df = pd.DataFrame(data=x_preprocessed,
                                         index=x.index,
                                         columns=feature_names)
        x_preprocessed_df.to_csv(os.path.join(output_path,
                                              os.path.join(intermediate_data_path, "preprocessed_features.csv")),
                                 sep=";")

        y_df = pd.DataFrame(data=y, columns=[label_name])
        y_df.to_csv(os.path.join(output_path,
                                 os.path.join(intermediate_data_path, "preprocessed_labels.csv")),
                    sep=";")

        # Feature selection for final model
        selected_indices_preprocessed, x_preprocessed, _ = mrmr_feature_selection(x_preprocessed,
                                                                                  y,
                                                                                  # score_func=f_classif,
                                                                                  num_features="log2n")

        feature_names_selected = feature_names[selected_indices_preprocessed]

        # Hyperparameter tuning for final model
        optimized_model = RandomizedSearchCV(model,
                                             param_distributions=clfs[clf]["parameters"],
                                             cv=10,
                                             random_state=seed)
        optimized_model.fit(x_preprocessed, y)

        # Add feature names enabling interpretability of output plots (only needed for some algorithms like ebm)
        try:
            model.set_params(**{"feature_names": feature_names_selected})
        except ValueError:
            pass

        # Save final model to file
        with open(os.path.join(model_result_path, f"{clf}_model.pickle"), "wb") as file:
            pickle.dump(optimized_model.best_estimator_, file)

        # Save hyperparameters of final model to file
        with open(os.path.join(model_result_path, f"{clf}_optimized_hyperparameters.json"), "w") as file:
            param_dict_json_conform = {}     # Necessary since JSON doesn't take some data types such as int32
            for key in optimized_model.best_params_:
                try:
                    param_dict_json_conform[key] = int(optimized_model.best_params_[key])
                except ValueError:
                    param_dict_json_conform[key] = optimized_model.best_params_[key]

            json.dump(param_dict_json_conform, file, indent=4)

        # Partial dependenced plots (DPD)
        for feature_index, _ in enumerate(np.arange(len(feature_names_selected))):
            if num_classes > 2:
                for target_index in np.arange(num_classes):
                    PartialDependenceDisplay.from_estimator(optimized_model.best_estimator_,
                                                            X=x_preprocessed,
                                                            features=[feature_index],
                                                            feature_names=feature_names_selected,
                                                            target=np.unique(y)[target_index])
                    plt.subplots_adjust(bottom=0.15)
                    plt.savefig(os.path.join(explainability_result_path,
            f"partial_dependence-{clf}_feature-{feature_names_selected[feature_index]}_class-{np.unique(y)[target_index]}.png"),
                                bbox_inches="tight")
                    plt.close()
            else:
                PartialDependenceDisplay.from_estimator(optimized_model.best_estimator_,
                                                        X=x_preprocessed,
                                                        features=[feature_index],
                                                        feature_names=feature_names_selected)
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(explainability_result_path,
                                         f"partial_dependence-{clf}_feature-{feature_names_selected[feature_index]}.png"),
                            bbox_inches="tight")
                plt.close()

        # SHAP analysis
        if verbose is True:
            print("[XAI] Computing SHAP importances")

        # Ensure plotting summary as bar for multiclass and beeswarm for binary classification
        if num_classes > 2:
            predictor = optimized_model.best_estimator_.predict_proba
        else:
            predictor = optimized_model.best_estimator_.predict

        explainer = shap.KernelExplainer(predictor, x_preprocessed)
        shap_values = explainer.shap_values(x_preprocessed)

        shap.summary_plot(shap_values=shap_values,
                          features=x_preprocessed,
                          feature_names=feature_names_selected,
                          class_names=optimized_model.best_estimator_.classes_,
                          show=False)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(explainability_result_path, f"shap_summary-{clf}.png"),
                    bbox_inches="tight")
        plt.close()

        # Get mean of feature importance scores and standard deviation over all folds
        overall_mean_importances_train = raw_importances_foldwise_mean_train / num_folds
        overall_std_importances_train = raw_importances_foldwise_std_train / num_folds
        overall_mean_importances_val = raw_importances_foldwise_mean_val / num_folds
        overall_std_importances_val = raw_importances_foldwise_std_val / num_folds

        # Plot feature importances as determined using training and validation data
        plot_title_permutation_importance = f"Permutation importance {clf} "
        plot_importances(importances_mean=overall_mean_importances_train,
                         importances_std=overall_std_importances_train,
                         feature_names=feature_names_selected,
                         plot_title=plot_title_permutation_importance + " - Training data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=False,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-train")
                         )

        plot_importances(importances_mean=overall_mean_importances_val,
                         importances_std=overall_std_importances_val,
                         feature_names=feature_names_selected,
                         plot_title=plot_title_permutation_importance + " - Validation data",
                         order_alphanumeric=True,
                         include_top=0,
                         display_plots=False,
                         save_path=os.path.join(explainability_result_path,
                                                plot_title_permutation_importance + "-test")
                         )

        # Calculate overall performance
        for metric in performance_foldwise:
            avg_metric = np.round(np.sum(performance_foldwise[metric]) / len(performance_foldwise[metric]), 2)
            clfs_performance[metric].append(avg_metric)

        if verbose is True:
            print(f"Finished {clf} with MCCV-wide confusion matrix:")
            print(cms)

        # Display and save confusion matrix figure
        seaborn.heatmap(cms, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("{} - Confusion matrix".format(clf))
        plt.savefig(os.path.join(performance_result_path, f"confusion_matrix-{clf}.png"))
        plt.close()

        # Save confusion matrix as CSV
        label_names = np.sort(np.unique(y))
        cm_df = pd.DataFrame(cms,
                             columns=[str(name) + " (actual)" for name in label_names],
                             index=[str(name) + " (predicted)" for name in label_names])
        cm_df.to_csv(os.path.join(performance_result_path, f"confusion_matrix-{clf}.csv"), sep=";")

    # Append performance to result table
    for metric in clfs_performance:
        results[metric] = clfs_performance[metric]

    # Save result table
    colors = ["dimgray", "gray", "darkgray", "maroon", "lightgray", "gainsboro"]
    results.to_csv(os.path.join(performance_result_path, "performances.csv"), sep=";")
    results.plot.bar(rot=45, color=colors).legend(loc="upper right")

    if verbose is True:
        print("[Results] Displaying performance")
        print(results)

    # Adjust legend position so it doesn't mask any bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, clfs_performance.keys(), loc="best", bbox_to_anchor=(1.13, 1.15))

    # Save and display performance plot
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(linestyle="dashed", axis="y")
    plt.title("Overall performance")
    plt.savefig(os.path.join(performance_result_path, "performance.png".format(clf)))
    plt.close()


if __name__ == "__main__":
    main()
