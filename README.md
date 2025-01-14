# Student Performance Prediction

This project aims to predict whether a student will pass or fail based on various features such as demographics, study time, and family background. 
Three machine learning models—K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM)—are trained and evaluated on a student performance dataset. 
The project includes data preprocessing, hyperparameter tuning, model evaluation, and visualization of results.

## Requirements
- Python 3.x
- Required libraries:
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - tabulate
    - os

 You can install the necessary libraries using the following command:
 
```bash
pip install pandas matplotlib seaborn scikit-learn tabulate
```

## Dataset
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

The dataset used is from the UCI Machine Learning Repository, specifically the Student Performance Data Set. 
This dataset contains attributes about students (e.g., demographics, study time, family support) and their final grades. 
The target variable (G3) indicates the final grade, where a grade of 10 or higher is considered a "pass" and below 10 is a "fail."

## Usage
### Step 1: Data Preprocessing
The dataset is first loaded into a Pandas DataFrame. Then, we create a new target variable pass_fail, where 1 indicates the student passed and 0 indicates the student failed. 
Categorical variables are converted into binary features using one-hot encoding.

### Step 2: Model Training
Three classification models are used for training:
- Optimized K-Nearest Neighbors (KNN): Hyperparameters (number of neighbors, distance metric, and weighting scheme) are tuned using GridSearchCV.
- Random Forest: A default Random Forest classifier is used.
- Support Vector Machine (SVM): A default SVM classifier is used with probability estimates enabled.

### Step 3: Model Evaluation
For each model, evaluation is done on the test set. The following metrics are reported:
- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- Log Loss

Cross-validation is also performed on each model to assess stability and generalization performance.

### Step 4: Results Visualization
The results are visualized as:
- A bar plot comparing model metrics (Accuracy, Precision, Recall, etc.).
- Confusion matrices for each model, showing the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
- ROC (Receiver Operating Characteristic) curves for each model, showing the trade-off between true positive rate and false positive rate.

All visualizations (model comparison bar chart, confusion matrices, and ROC curves) are saved in the 'Data Visualization/' directory.

## Results
The models are compared based on:
- **Accuracy:** Random Forest outperforms the other models with 91.1% accuracy.
- **Precision:** Random Forest also shows the highest precision, closely followed by SVM.
- **Recall:** Optimized KNN has the highest recall at 94.2%, but this comes at the cost of lower specificity.
- **F1-Score:** Random Forest achieves the highest F1-score, indicating the best balance between precision and recall.

### Cross-Validation Results
- **Random Forest** has the highest mean accuracy (92.1%) with the lowest variance (6.9%), making it the most stable model.
- **Optimized KNN** performs well in terms of recall but has a lower mean accuracy and higher variance.

## Conclusion
- **Random Forest** is the best-performing model overall, providing the highest accuracy, precision, and F1-score.
- **SVM** is a solid alternative, offering a good balance of precision and recall.
- **KNN** is optimized for recall but sacrifices accuracy and specificity.
