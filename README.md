# TabularClassificationTemplate

This project aims to create a template for solving classification problems for tabular data.
The template handles binary and multi-class problems. Among others, the project includes an exploratory data analysis, a preprocessing pipeline before train/test splitting, a foldw-ise preprocessing pipeline after train/test splitting, a scalable and robust Monte Carlo cross-validation scheme, six classification algorithms which are evaluated for four performance metrics and set set of capabilities enabling explainability including visualizations.

<img src="Assets/tct_flow_simple.png" alt="Workflow diagram" width="600"/>

Content:

- Exploratory data analysis via Pandas Profiling
- Preprocessing
    - Removing all-NA instances
    - Removing features with constant value over all instances (ignoring NaNs)
    - Removing features with too many missing values
- Fold-wise preprocessing
    - Normalization
    - Filling missing values using kNN imputation
    - Resampling for handling label imbalances via SMOTE
- Performance estimation using Monte Carlo cross validation with multiple metrics
    - Accuracy
    - AUC
    - Sensitivity / Recall
    - Specificity
- Feature selection using mRMR
- Hyperparameter optimization (Using random grid search)
- Training and evaluation of multiple classification algorithms
    - Explainable Boosting Machine
    - XGBoost
    - k-nearest Neighbors
    - Decision Tree
    - Random Forest
    - Neural Network
- Explainable Artificial Intelligence (XAI)
    - Permutation feature importance (+ visualizations)
    - Partial dependence plots
    - SHAP values (+ summary visualization)
- Visualization of performance evaluation
    - Performances for each classification model via barplot
    - Confusion matrices

- Outputs:
    - EDA: results as HTML report
    - Intermediate data: preprocessed data for final models as CSV
    - Models: pickled objects and tuned hyperparameters as JSON
    - Performance: Confusion matrices and overall performance metrics for each model as CSV and visalization
    - XAI: Partial dependence plots, permutation feature importances and SHAP summary plots
