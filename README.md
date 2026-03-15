# Student Performance Prediction

This project trains and compares three classifiers on the UCI Student Performance dataset:

- K-Nearest Neighbors (KNN), tuned with `GridSearchCV`
- `RandomForestClassifier`
- `SVC` with probability estimates enabled

The script handles dataset inspection, pass/fail target creation from the final grade, one-hot encoding of categorical features, train/test splitting, feature scaling, metric reporting, cross-validation, and plot generation.

## What the project does
`main.py` runs a full end-to-end classification workflow:

- Loads `student/student-mat.csv`
- Checks dataset shape, sample rows, duplicates, missing values, and dataframe info
- Creates a binary `pass_fail` target from `G3`
- One-hot encodes the categorical columns
- Splits the data into 80% training and 20% test sets
- Standardizes features before model fitting
- Tunes KNN hyperparameters with 10-fold stratified cross-validation
- Trains the three models
- Evaluates each model on the test set
- Prints a comparison table with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Specificity
  - Log-loss
- Prints a second table with cross-validation mean accuracy and standard deviation

It also saves visual output under `Data Visualization/`, including:

- A model-comparison bar chart
- A confusion matrix for KNN
- A confusion matrix for Random Forest
- A confusion matrix for SVM
- A ROC curve for KNN
- A ROC curve for Random Forest
- A ROC curve for SVM

## Project layout
`main.py` contains the full training, evaluation, and visualization pipeline.

`student/` contains the local copy of the dataset and related source files.

`Data Visualization/` contains generated plots from previous runs.

## Dataset
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

The dataset used is from the UCI Machine Learning Repository, specifically the Student Performance Data Set. 
This dataset contains attributes about students (e.g., demographics, study time, family support) and their final grades. 
The target variable (G3) indicates the final grade, where a grade of 10 or higher is considered a "pass" and below 10 is a "fail."

## Requirements
This project was run with Python 3.10 in a local virtual environment at `.venv`.

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to run
From the project root:

```bash
source .venv/bin/activate
python3 main.py
```

The script prints dataset inspection output, the selected KNN hyperparameters, per-model evaluation metrics, and a final cross-validation summary in the terminal.

## Results
Measured results from a fresh run on the local student dataset with `random_state=42`, an 80/20 train-test split, and `stratify=y`:

| Metric | Optimized KNN | Random Forest | SVM |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7089 | 0.8861 | 0.8101 |
| Precision | 0.7143 | 0.9583 | 0.8519 |
| Recall | 0.9434 | 0.8679 | 0.8679 |
| F1-score | 0.8130 | 0.9109 | 0.8598 |
| Specificity | 0.2308 | 0.9231 | 0.6923 |
| Log-loss | 0.4898 | 0.2998 | 0.3766 |

Cross-validation results from the same run:

| Model | CV Mean Accuracy | CV Std Accuracy |
| --- | ---: | ---: |
| Random Forest | 0.9149 | 0.0540 |
| SVM | 0.8417 | 0.0543 |
| Optimized KNN | 0.7438 | 0.0353 |

Best hyperparameters selected for KNN:

- `metric='manhattan'`
- `n_neighbors=11`
- `weights='distance'`

Generated visualization charts:

- `Data Visualization/model_comparison.png`
- `Data Visualization/knn_confusion_matrix.png`
- `Data Visualization/rf_confusion_matrix.png`
- `Data Visualization/svm_confusion_matrix.png`
- `Data Visualization/knn_roc.png`
- `Data Visualization/rf_roc.png`
- `Data Visualization/svm_roc.png`

## Notes
- The script currently evaluates all three models on the test set, not on both train and test sets.
- Feature scaling is applied to the entire processed feature set before fitting each model.
- The train/test split is now stratified to preserve the pass/fail class balance across both sets.
- KNN hyperparameter tuning now runs with `n_jobs=1`, which avoids multiprocessing failures in restricted environments.
