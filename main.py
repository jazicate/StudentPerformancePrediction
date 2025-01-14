import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, auc, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import os

# Create the folder for saving visualizations if it doesn't exist
if not os.path.exists('Data Visualization'):
    os.makedirs('Data Visualization')

# Import Dataset
try:
    df = pd.read_csv(filepath_or_buffer="student/student-mat.csv", sep=";")
except ImportError:
    print("Error: Import Dataset failed")
    exit()

# Data Checking
print("\n\n***Data Checking***")
print(f"Dataset Dimension: {df.shape}")
print(f"\nRandomly selected 5 instances:\n {df.sample(5)}")
print(f"\nLast 5 instances:\n {df.tail()}")
print("\nAny NaN values: Yes" if df.isna().values.any() else "Any NaN values: No")
print("Any Duplicate values: Yes" if df.duplicated().values.any() else "Any Duplicate values: No")
print("\n\n**Dataset info**")
print(df.info())

# Preprocess Dataset
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                       'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
df_binary = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df_binary.drop(columns=['G3', 'pass_fail'])
y = df_binary['pass_fail']

# Split data: training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tune hyperparameters for KNN using GridSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters from GridSearchCV
best_params = grid_search.best_params_
print(f"\nBest Parameters for KNN: {best_params}")

# Retrain KNN model with the best parameters
knn_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'],
                                weights=best_params['weights'])
knn_best.fit(X_train_scaled, y_train)

# Make predictions using optimized KNN
y_train_pred = knn_best.predict(X_train_scaled)
y_test_pred = knn_best.predict(X_test_scaled)


# Evaluate KNN performance
def evaluate_model(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    logloss = log_loss(y_true, knn_best.predict_proba(X_test_scaled))
    return {
        f'{model_name} Accuracy': accuracy,
        f'{model_name} Precision': precision,
        f'{model_name} Recall': recall,
        f'{model_name} F1-Score': f1,
        f'{model_name} Specificity': specificity,
        f'{model_name} Log Loss': logloss,
    }


knn_metrics = evaluate_model(y_test, y_test_pred, "Optimized KNN")

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Train and evaluate other models (Random Forest, SVM)
model_metrics = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    model_metrics[model_name] = evaluate_model(y_test, y_test_pred, model_name)

# Combine KNN metrics and model comparison
all_metrics = {**knn_metrics, **{k: v for model in model_metrics for k, v in model_metrics[model].items()}}

# Output Model Comparison Results
print("\n\n***Model Comparison Results***")
print(tabulate(all_metrics.items(), headers=['Metric', 'Value'], tablefmt='fancy_grid'))

# Perform cross-validation for model validation
validation_results = []
for model_name, model in models.items():
    cv_score = cross_val_score(model, X_train_scaled, y_train,
                               cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')
    validation_results.append({
        'Model': model_name,
        'CV Mean Accuracy': cv_score.mean(),
        'CV Std Accuracy': cv_score.std()
    })

# Cross-validation for optimized KNN
cv_score_knn = cross_val_score(knn_best, X_train_scaled, y_train,
                               cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')
validation_results.append({
    'Model': 'Optimized KNN',
    'CV Mean Accuracy': cv_score_knn.mean(),
    'CV Std Accuracy': cv_score_knn.std()
})

# Output Cross-Validation Results
validation_results_df = pd.DataFrame(validation_results)
print("\n\n***Model Validation Results***")
print(tabulate(validation_results_df, headers='keys', tablefmt='fancy_grid', showindex=False))


# Data Visualization: Model Comparison on Metrics
def plot_metrics_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=values)  # Removed the 'palette' argument
    plt.xticks(rotation=90)
    plt.title('Model Comparison Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig('Data Visualization/model_comparison.png')
    plt.close()


plot_metrics_comparison(all_metrics)


# Confusion Matrix for KNN, Random Forest, and SVM
def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)  # Corrected: passing predicted labels
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'Data Visualization/{filename}.png')
    plt.close()

# Corrected the confusion matrix call to pass predictions, not accuracy score
plot_confusion_matrix(y_test, y_test_pred, 'Optimized KNN', 'knn_confusion_matrix')
plot_confusion_matrix(y_test, models['Random Forest'].predict(X_test_scaled), 'Random Forest', 'rf_confusion_matrix')  # Corrected here
plot_confusion_matrix(y_test, models['SVM'].predict(X_test_scaled), 'SVM', 'svm_confusion_matrix')


# ROC Curves for KNN, Random Forest, and SVM
def plot_roc_curve(y_true, y_pred_proba, model_name, filename):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'Data Visualization/{filename}_roc.png')
    plt.close()


plot_roc_curve(y_test, knn_best.predict_proba(X_test_scaled), 'Optimized KNN', 'knn')
plot_roc_curve(y_test, models['Random Forest'].predict_proba(X_test_scaled), 'Random Forest', 'rf')
plot_roc_curve(y_test, models['SVM'].predict_proba(X_test_scaled), 'SVM', 'svm')
