# TabularClassificationTemplate

This project aims to create a template for solving classification problems using Scikit-Learn for
tabular data.
The template shall handle binary as well as multi-class classification problems and shall include
a preprocessing pipeline. Further, the template shall be easily adaptable and extendible for a
simple integration into larger machine learning workflows.

<!---
<img src="Assets/tct_workflow_details.png" alt="Workflow diagram" width="600"/>
-->

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
    - k-nearest Neighbors
    - Decision Tree
    - Random Forest
    - Neural Network
- Explainable Artificial Intelligence (XAI)
    - Permutation feature importance
    - Partial dependence plots
    - SHAP summary plot
- Visualization of performance evaluation
    - Performances for each classification model via barplot
    - Confusion matrices

Outputs:
    - EDA: results as HTML report
    - Intermediate data: preprocessed data for final models as CSV
    - Models: pickled objects and tuned hyperparameters as JSON
    - Performance: Confusion matrices and overall performance metrics for each model as CSV and visalization
    - XAI: Partial dependence plots, permutation feature importances and SHAP summary plots
